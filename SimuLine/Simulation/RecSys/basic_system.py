import torch
import numpy as np
from tqdm import tqdm
import dgl
from bpemb import BPEmb
from logging import getLogger
from recbole.utils import set_color
from torch.nn.functional import normalize

EPSILONG = 1e-8
EPSILONG_U = 1e2


class BasicSystem:
    def __init__(self, config, metric, name='SimuLine-BasicSystem'):
        self._name = name
        self._config = config
        self._metric = metric
        self.update_promotion_details()
        assert self._config['num_from_match'] + self._config['num_from_cold_start'] + self._config['num_from_hot'] + self._config['num_from_promote'] == self._config['recommendation_list_length']
        self._logger = getLogger()
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))
    
    def update_promotion_details(self):
        if self._config['promote_type'] == 'content':
            bpemb = BPEmb(lang="no", dim=100, vs=200000)
            bpemb_vectors = torch.Tensor(bpemb.vectors)
            bpemb_vectors_lower = bpemb_vectors.min(0).values
            bpemb_vectors_higher = bpemb_vectors.max(0).values
            random_content_base = torch.rand(bpemb_vectors.shape[1])  # 多个topic很美好，但是计算太复杂，算逑
            self._promote_content = random_content_base * (bpemb_vectors_higher - bpemb_vectors_lower) + bpemb_vectors_lower
        elif self._config['promote_type'] == 'author':
            num_creator = self._config['num_creator']
            self._promote_author = torch.Tensor(sorted(np.random.choice(a=num_creator, size=self._config['num_from_promote'], replace=False, p=None))).int()
        else:
            raise Exception('Unexpected Promote Type')
    
    def mount_model(self, model, user_id_field, item_id_field, user_id_mapping, item_id_mapping, device, batch_size):
        self._model = model
        self._user_id_field = user_id_field
        self._item_id_field = item_id_field
        self._user_id_mapping = user_id_mapping
        self._item_id_mapping = item_id_mapping
        self._device = device
        self._batch_size = batch_size
    
    def get_match_matrix(self, num_user, num_active_item):
        num_embedded_user = len(self._user_id_mapping) - 1
        num_embedded_item = len(self._item_id_mapping) - 1
        inter_user = torch.Tensor(list(range(1, num_embedded_user+1))).type(torch.int64).unsqueeze(1).expand(num_embedded_user, num_embedded_item).reshape(-1)
        inter_item = torch.Tensor(list(range(1, num_embedded_item+1))).type(torch.int64).unsqueeze(0).expand(num_embedded_user, num_embedded_item).reshape(-1)
        num_batch = int(num_embedded_user * num_embedded_item / self._batch_size)
        inter_pred_list = []
        # Scoring using batch
        # Only user&item with trained embedding get envolved
        # So this is a sub-matrix
        self.log_info('Part one: from match - Scoring')
        for i in range(num_batch+1):
            start = self._batch_size * i
            if i == num_batch:
                end = num_embedded_user * num_embedded_item
            else:
                end = self._batch_size * (i+1)
            pred = self._model.predict({
                self._user_id_field: inter_user[start:end].to(self._device),
                self._item_id_field: inter_item[start:end].to(self._device),
            })
            inter_pred_list.append(pred.detach().cpu())
        inter_pred = torch.cat(inter_pred_list, dim=0)
        inter_user_index = torch.from_numpy(self._user_id_mapping[1:].astype(np.int64)).unsqueeze(1).expand(num_embedded_user, num_embedded_item).reshape(-1)
        inter_item_index = torch.from_numpy(self._item_id_mapping[1:].astype(np.int64)).unsqueeze(0).expand(num_embedded_user, num_embedded_item).reshape(-1)
        inter_user_item = torch.cat([inter_user_index.unsqueeze(0), inter_item_index.unsqueeze(0)]).type(torch.int64)
        # The Full-Score-Matrix & Add random noise with a small value as blank scores
        match_matrix = torch.sparse.FloatTensor(inter_user_item, inter_pred, [num_user, num_active_item]).to_dense() 
        match_matrix += torch.normal(torch.zeros(match_matrix.shape), torch.ones(match_matrix.shape) * EPSILONG)
        return match_matrix
    
    def recommend(self, graph_dataset):
        # 0. Part zero: Prepare
        self.log_info('Part zero: prepare')
        num_user, num_item = graph_dataset.num_node['user'], graph_dataset.num_node['article']
        total_num_create_each_round = self._config['num_creator'] * self._config['num_create']
        earliest_active_round = max(0, graph_dataset._round - self._config['n_round'])
        item_index_padding = earliest_active_round * total_num_create_each_round  # articles with index before this are unactivited
        num_active_item = num_item - item_index_padding - total_num_create_each_round  # exclude newly created articles
        sg_recent_N_round_article = dgl.node_subgraph(graph_dataset.graph, {
            'user': torch.Tensor(range(graph_dataset.num_node['user'])).int(),  # all user
            'article': torch.Tensor(range(item_index_padding, item_index_padding + num_active_item)).int(),  # active articles
            'creator': torch.Tensor(range(graph_dataset.num_node['creator'])).int(),  # all creator
        })  # whether store_id makes no different
        exposure_matrix = sg_recent_N_round_article.adj(etype='exposure').to_dense()
        recommendation_lists = []

        # 1. Part one: from hot
        # recommend the hotest N articles in N round counted with self._config['hot_inter_type']
        # With a higher priority than match and promote
        # Hot -> match -> promote
        # Cold start is independent to this system 
        self.log_info('Part one: from hot')
        if self._config['num_from_hot'] > 0:
            inter_type = self._config['hot_inter_type']
            item_degree = sg_recent_N_round_article.in_degrees(etype=inter_type).expand(num_user, num_active_item)
            # set exposed value to "0"
            item_degree = item_degree * (1 - exposure_matrix)  # filter exposed
            hotest = torch.topk(item_degree, self._config['num_from_hot'], dim=1, largest=True, sorted=True)
            recommendation_from_hot = hotest.indices + item_index_padding
            user_index = torch.Tensor(range(self._config['num_user'])).int().expand(self._config['num_from_hot'], self._config['num_user']).permute(1, 0)
            exposure_matrix += torch.sparse.FloatTensor(torch.cat([user_index.reshape(1, -1), hotest.indices.reshape(1, -1)], dim=0), torch.ones([self._config['num_user'] * self._config['num_from_hot']]).int(), [num_user, num_active_item]).to_dense() 
            recommendation_lists.append(recommendation_from_hot)
        else:
            pass
        
        # 2. Part two: from match
        self.log_info('Part two: from match')
        # match_matrix: have embeddings -> score; any one doesn't -> random small value
        # priority with trained, sub with random
        match_matrix = self.get_match_matrix(num_user, num_active_item)
        # rank exposed item to the tail via reduce the value
        match_matrix = match_matrix * (1 - exposure_matrix)
        match_matrix = match_matrix + exposure_matrix * match_matrix.min()
        top_matched = torch.topk(match_matrix, self._config['num_from_match'], dim=1, largest=True, sorted=True)
        recommendation_from_match = top_matched.indices + item_index_padding
        user_index = torch.Tensor(range(self._config['num_user'])).int().expand(self._config['num_from_match'], self._config['num_user']).permute(1, 0)
        exposure_matrix += torch.sparse.FloatTensor(torch.cat([user_index.reshape(1, -1), top_matched.indices.reshape(1, -1)], dim=0), torch.ones([self._config['num_user'] * self._config['num_from_match']]).int(), [num_user, num_active_item]).to_dense()
        recommendation_lists.append(recommendation_from_match)

        # 3. Part three: from cold start
        self.log_info('Part three: from cold start')
        inter_type = self._config['cold_start_inter_type']
        if inter_type == 'random':
            candidates = list(range(total_num_create_each_round))
            random_recommendation = []
            for i in range(graph_dataset._num_node['user']):
                random_recommendation.append(np.random.choice(candidates, self._config['num_from_cold_start'], replace=False))
            recommendation_from_cold_start = torch.Tensor(random_recommendation).int() + (num_item - total_num_create_each_round)
            recommendation_lists.append(recommendation_from_cold_start)
        else:
            user_history_intered_author = torch.matmul(
                sg_recent_N_round_article.adj(etype=inter_type).to_dense(), 
                sg_recent_N_round_article.adj(etype='create').to_dense().T
            )  # mm(inter_matrix, history_create_matrix)
            sg_new_article = dgl.node_subgraph(graph_dataset.graph, {
                'user': torch.Tensor(range(graph_dataset.num_node['user'])).int(),  # all user
                'article': torch.Tensor(range(num_item - total_num_create_each_round, num_item)).int(),  # active articles
                'creator': torch.Tensor(range(graph_dataset.num_node['creator'])).int(),  # all creator
            })
            newly_create_matrix = sg_new_article.adj(etype='create')
            user_history_intered_author_new_article = torch.matmul(user_history_intered_author, newly_create_matrix.to_dense())
            recommendation_from_cold_start = torch.multinomial(user_history_intered_author_new_article + EPSILONG, self._config['num_from_cold_start']) 
            recommendation_from_cold_start = recommendation_from_cold_start + (num_item - total_num_create_each_round)
            recommendation_lists.append(recommendation_from_cold_start)
            # Different source, no need to update exposure_matrix

        # 4. Part four: from promote
        self.log_info('Part four: from promote')
        if self._config['num_from_promote'] > 0:
            # Check Modify
            if self._config['promote_type'] == 'content':
                if self._promote_content.shape[0] != self._config['num_from_promote']:
                    self.update_promotion_details()
            elif self._config['promote_type'] == 'author':
                if self._promote_author.shape[0] != self._config['num_from_promote']:
                    self.update_promotion_details()
            else:
                raise Exception('Unexpected Promote Type')
            # Rec
            if self._config['promote_type'] == 'content':
                active_article_latent = sg_recent_N_round_article.nodes['article'].data['latent']
                promote_score = torch.matmul(normalize(active_article_latent, p=2, dim=-1).float(), normalize(self._promote_content.unsqueeze(1), p=2, dim=0)).squeeze(1)
                promote_score_topic_value = promote_score.expand(num_user, num_active_item)
                # set exposed item to "-EPSILONG_U" -> softmax=0 
                promote_score_topic_value = (promote_score_topic_value + EPSILONG_U) * (1 - exposure_matrix) - EPSILONG_U  # filter exposed
                promote_score_topic_value = promote_score_topic_value.softmax(dim=-1)
                recommendation_from_promote = torch.multinomial(promote_score_topic_value + EPSILONG, self._config['num_from_promote']) + item_index_padding
                # the last one, no need to update exposure_matrix
            elif self._config['promote_type'] == 'author':
                # item_index_padding + round_padding + position_padding
                promote_author_article_index = []
                num_active_round = min(graph_dataset._round, self._config['n_round'])
                for author_id in self._promote_author.tolist():
                    author_articles = []
                    for rnd in range(num_active_round):
                        round_padding = rnd * total_num_create_each_round
                        for i in range(self._config['num_create']):
                            position_padding = author_id * self._config['num_create'] + i
                            author_articles.append(item_index_padding + round_padding + position_padding)
                    promote_author_article_index.append(author_articles)
                promote_author_article_index = torch.Tensor(promote_author_article_index).long()  # abs index
                # 下面这个不能保证promotion pool里的作者都被曝光
                promote_matrix = 1 - exposure_matrix[:, (promote_author_article_index - item_index_padding).reshape(-1)]
                promote_matrix /= promote_matrix.sum(-1).expand(promote_matrix.shape[1], promote_matrix.shape[0]).permute(1, 0)
                promote_article_index = torch.multinomial(promote_matrix, self._config['num_from_promote'])
                recommendation_from_promote = promote_author_article_index.reshape(-1)[promote_article_index.reshape(-1)].reshape(promote_article_index.shape)
            else:
                raise Exception('Unexpected Promote Type')
            if graph_dataset._round % self._config['promote_round'] == 0:
                self.update_promotion_details()
            recommendation_lists.append(recommendation_from_promote)
        else:
            pass
        
        # 5. Merge
        recommendation_list = torch.cat(recommendation_lists, dim=1)
        
        return recommendation_list


