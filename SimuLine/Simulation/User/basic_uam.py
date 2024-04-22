import torch

from logging import getLogger
from recbole.utils import set_color
from torch.nn.functional import normalize

EPSILON = 1e-5

class BasicUAM:
    def __init__(self, config, metric, name='SimuLine-BasicUAM'):
        self._name = name
        self._config = config
        self._metric = metric
        self._logger = getLogger()

    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))

    def action(self, graph_dataset, recommendations):
        # 0. Part zero: Prepare
        self.log_info('Part zero: prepare') 
        total_num_create_each_round = self._config['num_creator'] * self._config['num_create']
        earliest_active_round = max(0, graph_dataset._round - self._config['n_round'])
        item_index_padding = earliest_active_round * total_num_create_each_round  # articles with index before this are unactivited
        user_latent = graph_dataset.graph.nodes['user'].data['latent']
        user_thres = graph_dataset.graph.nodes['user'].data['threshold']
        user_like_quality_weight = graph_dataset.graph.nodes['user'].data['like_quality_weight']
        user_like_match_weight = graph_dataset.graph.nodes['user'].data['like_match_weight']
        user_concentration = graph_dataset.graph.nodes['user'].data['concentration']
        item_latent = graph_dataset.graph.nodes['article'].data['latent'][item_index_padding:]
        item_quality = graph_dataset.graph.nodes['article'].data['quality'][item_index_padding:]
        best_stable_quality = self._config['best_stable_quality']
        latent_dim = user_latent.shape[1]
        
        # score
        self.log_info('Part one: score')
        recommendation_latent = item_latent[(recommendations - item_index_padding).reshape(-1).type(torch.int64)].reshape(-1, self._config['recommendation_list_length'], latent_dim)
        latent_match_score = torch.matmul(normalize(user_latent, p=2, dim=-1).unsqueeze(1), normalize(recommendation_latent, p=2, dim=-1).permute(0, 2, 1)).squeeze(1)
        user_concentration_ = user_concentration.unsqueeze(1).expand(latent_match_score.shape)
        latent_match_score_softmaxed = (user_concentration_ * latent_match_score).softmax(dim=1)
        
        # click (Mat)
        self.log_info('Part two: click')
        click = torch.multinomial(latent_match_score_softmaxed, num_samples=self._config['num_click'], replacement=False)
        click_index = click.reshape(-1)
        user_index = [torch.Tensor(list(range(len(click)))).type(torch.int64).unsqueeze(1) for i in range(self._config['num_click'])]
        user_index = torch.cat(user_index, dim=1).reshape(-1)
        clicked_id = recommendations[user_index, click_index]
        clicked_id = clicked_id.reshape(-1, self._config['num_click'])
        
        # like (Pair)
        self.log_info('Part three: like')
        clicked_quality = item_quality[(clicked_id - item_index_padding).reshape(-1).type(torch.int64)].reshape(clicked_id.shape)
        clicked_latent = item_latent[(clicked_id - item_index_padding).reshape(-1).type(torch.int64)].reshape([clicked_id.shape[0], clicked_id.shape[1], item_latent.shape[-1]])
        clicked_match = latent_match_score[user_index, click_index].reshape(-1, self._config['num_click'])
        user_like_quality_weight_ = user_like_quality_weight.unsqueeze(1).expand(clicked_quality.shape)
        user_like_match_weight_ = user_like_match_weight.unsqueeze(1).expand(clicked_quality.shape)
        clicked_utility = user_like_quality_weight_ * (clicked_quality / best_stable_quality) + user_like_match_weight_ * clicked_match
        
        # filter liked items
        user_thres = user_thres.unsqueeze(1).expand(clicked_quality.shape)
        clicked_is_liked = (clicked_utility > user_thres).type(torch.int32)
        liked_id = (clicked_id * clicked_is_liked).to_sparse()
        liked_id = torch.cat([liked_id.indices(), liked_id.values().unsqueeze(0)], dim=0)[[0, 2]].T.type(torch.int32)
        
        self._metric.user_record['liked_quality_contribution'].append(user_like_quality_weight[liked_id[:,0].long()] * (item_quality[liked_id[:,1].long() - item_index_padding] / best_stable_quality))
        self._metric.user_record['liked_match_contribution'].append(user_like_match_weight[liked_id[:,0].long()] * (normalize(user_latent[liked_id[:,0].long()], p=2, dim=-1) * normalize(item_latent[liked_id[:,1].long() - item_index_padding], p=2, dim=-1)).sum(-1))

        # 「 Abort 」Update user_concentration
        # user_like_ratio = clicked_is_liked.sum(dim=1) / self._config['num_click']
        # graph_dataset.graph.nodes['user'].data['concentration'] = (1 - user_like_ratio).clip(min=1e-8, max=1-1e-8)
        # user_concentration = graph_dataset.graph.nodes['user'].data['concentration']
        
        # User interest shift
        self.log_info('Part four: user interest shift')
        # Step Direction and Scale
        shift_directions = clicked_latent - user_latent[user_index].reshape(-1, self._config['num_click'], latent_dim)
        shift_directions /= shift_directions.norm(dim=2).unsqueeze(2).expand(shift_directions.shape)
        shift_directions *= self._config['uam_delta']
        shift_directions *= ((clicked_is_liked * 2) - 1).unsqueeze(2).expand(shift_directions.shape)
        
        # Shift Prob
        user_concentration_ = user_concentration.unsqueeze(1).expand(clicked_utility.shape)
        user_shift_prob = ((-1 * user_concentration_) / (clicked_utility.clip(min=0) + EPSILON)).exp()
        # higher concentration -> lower prob
        # user_do_shift = (torch.rand(user_shift_prob.shape) <= user_shift_prob).int()
        # shift_step = (shift_directions.permute(2,0,1) * user_do_shift * user_concentration_).permute(1,2,0).sum(dim=1)
        shift_step = (shift_directions.permute(2,0,1) * user_shift_prob).permute(1,2,0).sum(dim=1)
        graph_dataset.graph.nodes['user'].data['latent'] += shift_step
        
        return clicked_id, liked_id

