import os
import json
import torch
import numpy as np
import pandas as pd
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info
from torch.nn.functional import normalize

from SimuLine.Simulation.Global.global_service import GlobalVariance
from SimuLine.Simulation.Graph.dgl_function import edges_in_prev_N_round

from logging import getLogger
from recbole.utils import set_color


class SimuLineGraphDataset(DGLDataset):
    """
    SimuLine DGL-Graph Constructor v2
    """
    def __init__(self, config, metric, name='SimuLine-SimuLineGraphDataset', hash_key='SimuLine DGL-Graph Constructor v2', force_reload=False):
        self._name = name
        self._hash_key = hash_key
        self._config = config
        self._metric = metric
        self._logger = getLogger()
        self._raw_dir = './SimuLine/Preprocessing/Data'
        self._save_dir = './SimuLine/Simulation/Data'
        self._num_node = {
            'user': config['num_user'],
            'article': config['num_creator'] * config['num_create'],
            'creator': config['num_creator'],
        }
        self._graph = None  # Save&Load
        self._reverse_etypes = None  # Save&Load
        self._graph_path = f"{self._save_dir}/graph_{self._config['version']}.json"
        self._save_info = f"{self._save_dir}/info_{self._config['version']}.json"
        self._round = 0
        # Pass additional information to DGL functions
        self._global_variance = GlobalVariance()
        self._global_variance.set_value('N_ROUND', self._config['n_round'])
        self._global_variance.set_value('ROUND', self._round)
        self._global_variance.write_file()
        # new articles without any exposure, k: creator_id, v: article_id
        self._cold_start_articles = {}
        # Non-using
        self._force_reload = force_reload
        self._verbose = True
        self._url = None
        self._load()
        self._metric.static_record['user_threshold'] = self._graph.nodes['user'].data['threshold']
        self._metric.static_record['user_like_quality_weight'] = self._graph.nodes['user'].data['like_quality_weight']
        self._metric.static_record['user_like_match_weight'] = self._graph.nodes['user'].data['like_match_weight']
        self._metric.static_record['user_concentration'] = self._graph.nodes['user'].data['concentration']
        self._metric.static_record['creator_concentration'] = self._graph.nodes['creator'].data['concentration']

    def process(self):
        self.log_info('Init Node Attributes')
        _attr = {node_type: {} for node_type in self._num_node}
        _attr['article']['quality'] = torch.ones([self._num_node['article']]) * self._config['init_quality_mean']
        # _attr['article']['quality'] = torch.normal(_attr['article']['quality'], torch.ones(_attr['article']['quality'].shape) * self._config['init_quality_var']).clip(min=1e-8)
        _attr['user']['threshold'] = torch.ones([self._num_node['user']]) * self._config['init_threshold_mean']
        _attr['user']['threshold'] = torch.normal(_attr['user']['threshold'], torch.ones(_attr['user']['threshold'].shape) * self._config['init_threshold_var']).clip(min=1e-8, max=1-1e-8)
        _attr['user']['like_quality_weight'] = torch.ones([self._num_node['user']]) * self._config['init_like_quality_weight_mean']
        _attr['user']['like_quality_weight'] = torch.normal(_attr['user']['like_quality_weight'], torch.ones(_attr['user']['like_quality_weight'].shape) * self._config['init_like_quality_weight_var']).clip(min=1e-8, max=1-1e-8)
        _attr['user']['like_match_weight'] = 1 - _attr['user']['like_quality_weight']
        _attr['user']['concentration'] = torch.ones([self._num_node['user']]) * self._config['init_user_concentration_mean']
        _attr['user']['concentration'] = torch.normal(_attr['user']['concentration'], torch.ones(_attr['user']['concentration'].shape) * self._config['init_user_concentration_var']).clip(min=1e-8, max=1-1e-8)
        _attr['creator']['concentration'] = torch.ones([self._num_node['creator']]) * self._config['init_creator_concentration_mean']
        _attr['creator']['concentration'] = torch.normal(_attr['creator']['concentration'], torch.ones(_attr['creator']['concentration'].shape) * self._config['init_creator_concentration_var']).clip(min=1e-8, max=1-1e-8)
        
        self.log_info('Init Node Latent')
        user_latent = np.load(f"{self._raw_dir}/user_sample.npy")  # keep
        creator_latent = np.load(f"{self._raw_dir}/creator_sample.npy")  # generate
        article_latent_mean = torch.from_numpy(creator_latent).unsqueeze(1).expand([self._num_node['creator'], self._config['num_create'], creator_latent.shape[-1]]).reshape(self._num_node['creator'] * self._config['num_create'], creator_latent.shape[-1])
        creator_concentration_ = _attr['creator']['concentration'].unsqueeze(1).unsqueeze(1).expand(self._num_node['creator'], self._config['num_create'], creator_latent.shape[-1]).reshape(self._num_node['creator'] * self._config['num_create'], creator_latent.shape[-1])
        article_latent = torch.normal(mean=article_latent_mean, std=creator_concentration_ * self._config['cam_delta'])
        _attr['user']['latent'] = torch.from_numpy(user_latent)
        _attr['article']['latent'] = article_latent
        _attr['creator']['latent'] = torch.from_numpy(creator_latent)
        
        self.log_info('Init Create Links')
        article_creator = torch.Tensor(list(range(self._num_node['creator']))).unsqueeze(1).expand([self._num_node['creator'], self._config['num_create']]).reshape([-1]).type(torch.int32).numpy().tolist()
        creator_article_links = np.array([[cid, iid] for iid, cid in enumerate(article_creator)])
        
        self.log_info('Regist Links')
        _links = {
            ('user', 'like', 'article'): [],
            ('user', 'click', 'article'): [],
            ('user', 'exposure', 'article'): [],
            ('creator', 'create', 'article'): creator_article_links.tolist(),
        }
        self._reverse_etypes = {}  # add reverse, need save
        cur_relations = list(_links.keys())
        for relation in cur_relations:
            reverse_relation = '{}_r'.format(relation[1])
            _links[(relation[2], reverse_relation, relation[0])] = [[p[1], p[0]] for p in _links[relation]]
            self._reverse_etypes[relation[1]] = reverse_relation
            self._reverse_etypes[reverse_relation] = relation[1]
        for node_type in self._num_node:  # add self loop
            node_ids = list(range(self._num_node[node_type]))
            self_link = np.array([node_ids, node_ids]).T.tolist()
            _links[(node_type, f"{node_type}_selfloop", node_type)] = self_link
        
        self.log_info('Build Graph')
        self._graph = dgl.heterograph(_links, num_nodes_dict=self._num_node, idtype=torch.int32)
        
        self.log_info('Mount Node Attributes')
        for ntype in _attr:
            for attr_type in _attr[ntype]:
                self._graph.nodes[ntype].data[attr_type] = _attr[ntype][attr_type]
        
        # Init Exposure & Click & Like
        self.init_exposure_click_like()

    def init_exposure_click_like(self):
        user_thres = self._graph.nodes['user'].data['threshold']
        user_like_quality_weight = self._graph.nodes['user'].data['like_quality_weight']
        user_like_match_weight = self._graph.nodes['user'].data['like_match_weight']
        user_concentration = self._graph.nodes['user'].data['concentration']
        user_latent = self._graph.nodes['user'].data['latent']
        item_quality = self._graph.nodes['article'].data['quality']
        item_latent = self._graph.nodes['article'].data['latent']
        best_stable_quality = self._config['best_stable_quality']

        self.log_info('Init Exposure&Click Links')
        candidates = list(range(self._num_node['article']))
        random_recommendation = []
        for i in range(self._num_node['user']):
            random_recommendation.append(np.random.choice(candidates, self._config['recommendation_list_length'], replace=False))
        random_recommendation = torch.Tensor(random_recommendation).int()
        # random_recommendation = torch.randint(low=0, high=self._num_node['article'], size=(self._num_node['user'], self._config['recommendation_list_length']))

        recommendation_latent = item_latent[random_recommendation.reshape(-1).type(torch.int64)].reshape(self._num_node['user'], self._config['recommendation_list_length'], -1)
        latent_match_score = torch.matmul(normalize(user_latent, p=2, dim=-1).unsqueeze(1), normalize(recommendation_latent, p=2, dim=-1).permute(0, 2, 1)).squeeze(1)
        user_concentration_ = user_concentration.unsqueeze(1).expand(latent_match_score.shape)
        latent_match_score_softmaxed = (user_concentration_ * latent_match_score).softmax(dim=1)
        click = torch.multinomial(latent_match_score_softmaxed, num_samples=self._config['num_click'], replacement=False)
        click_index = click.reshape(-1)
        click_user_index = torch.Tensor(list(range(self._num_node['user']))).unsqueeze(1).expand(self._num_node['user'], self._config['num_click']).reshape(-1)
        click_article_index = random_recommendation[click_user_index.type(torch.int64), click_index.type(torch.int64)]
        exposure_user_index = torch.Tensor(list(range(self._num_node['user']))).unsqueeze(1).expand(self._num_node['user'], self._config['recommendation_list_length']).reshape(-1)
        exposure_article_index = random_recommendation.reshape(-1)
        # add exposure links to dgl-graph
        users = exposure_user_index.type(torch.int32)
        articles = exposure_article_index.type(torch.int32)
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='exposure',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='exposure_r',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )
        # add click links to dgl-graph
        users = click_user_index.type(torch.int32)
        articles = click_article_index.type(torch.int32)
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='click',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='click_r',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )

        self.log_info('Init Like Links')
        # computing utility
        click = self._graph.find_edges(torch.Tensor(list(range(self._graph.num_edges('click')))).type(torch.int32), 'click')
        clicked_id = click[1].reshape(self._num_node['user'], -1)
        clicked_quality = item_quality[clicked_id.reshape(-1).type(torch.int64)].reshape(clicked_id.shape)
        clicked_latent = item_latent[clicked_id.reshape(-1).type(torch.int64)].reshape([clicked_id.shape[0], clicked_id.shape[1], item_latent.shape[-1]])
        clicked_match = torch.matmul(normalize(clicked_latent, p=2, dim=-1), normalize(user_latent, p=2, dim=-1).unsqueeze(-1)).squeeze(-1)
        user_like_quality_weight = user_like_quality_weight.unsqueeze(1).expand(clicked_quality.shape)
        user_like_match_weight = user_like_match_weight.unsqueeze(1).expand(clicked_quality.shape)
        clicked_utility = user_like_quality_weight * (clicked_quality / best_stable_quality) + user_like_match_weight * clicked_match
        # filter liked items
        user_thres = user_thres.unsqueeze(1).expand(clicked_quality.shape)
        clicked_is_liked = (clicked_utility > user_thres).type(torch.int32)
        liked_id = (clicked_id * clicked_is_liked).to_sparse()
        liked_id = torch.cat([liked_id.indices(), liked_id.values().unsqueeze(0)], dim=0)[[0, 2]].T.type(torch.int32)
        # add like links to dgl-graph
        users = liked_id[:, 0]
        articles = liked_id[:, 1]
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='like',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='like_r',
            data={
                'round': torch.zeros([users.shape[0]])
            }
        )
    
    
    def save(self):
        r"""
        保存图和标签
        """
        self.log_info('Saving to cache')
        save_graphs(self._graph_path, [self._graph])
        # 在Python字典里保存其他信息
        save_info(self._save_info, {
            'reverse_etypes': json.dumps(self._reverse_etypes),
        })

    def load(self):
        r"""
         从目录 `self.save_path` 里读取处理过的数据
        """
        self.log_info('Loading from cache')
        graphs, label_dict = load_graphs(self._graph_path)
        self._graph = graphs[0]
        info = load_info(self._save_info)
        self._reverse_etypes = json.loads(info['reverse_etypes'])

    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        return os.path.exists(self._graph_path) and os.path.exists(self._save_info)

    @property
    def num_node(self):
        return self._num_node

    @property
    def graph(self):
        return self._graph
    
    @property
    def round(self):
        return self._round
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))
    
    def sync_round(self):
        self._global_variance.set_value('ROUND', self._round)
        self._global_variance.write_file()
    
    def interaction_records(self):
        self.sync_round()
        # filter active nodes
        total_num_create_each_round = self._config['num_creator'] * self._config['num_create']
        earliest_active_round = max(0, self._round - self._config['n_round'])
        item_index_padding = earliest_active_round * total_num_create_each_round  # articles with index before this are unactivited
        num_active_item = self.num_node['article'] - item_index_padding - total_num_create_each_round  # exclude newly created articles
        sg_recent_N_round_article = dgl.node_subgraph(self.graph, {
            'user': torch.Tensor(range(self.num_node['user'])).int(),  # all user
            'article': torch.Tensor(range(item_index_padding, item_index_padding + num_active_item)).int(),  # active articles
            'creator': torch.Tensor(range(self.num_node['creator'])).int(),  # all creator
        })

        # query data
        like_eid = sg_recent_N_round_article.filter_edges(edges_in_prev_N_round, etype='like')
        click_eid = sg_recent_N_round_article.filter_edges(edges_in_prev_N_round, etype='click')
        exposure_eid = sg_recent_N_round_article.filter_edges(edges_in_prev_N_round, etype='exposure')
        like_round = sg_recent_N_round_article.edges['like'].data['round'][like_eid]
        click_round = sg_recent_N_round_article.edges['click'].data['round'][click_eid]
        exposure_round = sg_recent_N_round_article.edges['exposure'].data['round'][exposure_eid]
        like = sg_recent_N_round_article.find_edges(like_eid.type(torch.int32), 'like')
        click = sg_recent_N_round_article.find_edges(click_eid.type(torch.int32), 'click')
        exposure = sg_recent_N_round_article.find_edges(exposure_eid.type(torch.int32), 'exposure')
        like = torch.cat([like[0].unsqueeze(0), like[1].unsqueeze(0), like_round.unsqueeze(0)]).T.type(torch.int32)
        click = torch.cat([click[0].unsqueeze(0), click[1].unsqueeze(0), click_round.unsqueeze(0)]).T.type(torch.int32)
        exposure = torch.cat([exposure[0].unsqueeze(0), exposure[1].unsqueeze(0), exposure_round.unsqueeze(0)]).T.type(torch.int32)
        return [like, click, exposure]

    # def update_embedding(self, user_embedding, item_embedding, user_id_mapping, item_id_mapping):
    #     num_user = self._num_node['user']
    #     num_article = self._num_node['article']
    #     user_emb = torch.zeros([num_user, user_embedding.shape[1]])
    #     item_emb = torch.zeros([num_article, item_embedding.shape[1]])
    #     user_emb[user_id_mapping.type(torch.long)] = user_embedding
    #     item_emb[item_id_mapping.type(torch.long)] = item_embedding
    #     self._graph.nodes['user'].data['embedding'] = user_emb
    #     self._graph.nodes['article'].data['embedding'] = item_emb

    def update_exposure(self, exposure):
        exposure_pair = torch.Tensor([[[uid, iid] for iid in rec_list] for uid, rec_list in enumerate(exposure)]).type(torch.int32)
        exposure_pair = exposure_pair.reshape(-1, 2)
        users = exposure_pair[:, 0]
        articles = exposure_pair[:, 1]
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='exposure',
            data={
                'round': self._round * torch.ones([exposure_pair.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='exposure_r',
            data={
                'round': self._round * torch.ones([exposure_pair.shape[0]])
            }
        )
    
    def update_consumption(self, click, like):
        click_pair = torch.Tensor([[[uid, iid] for iid in click_list] for uid, click_list in enumerate(click)]).type(torch.int32)
        click_pair = click_pair.reshape(-1, 2)
        users = click_pair[:, 0]
        articles = click_pair[:, 1]
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='click',
            data={
                'round': self._round * torch.ones([click_pair.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='click_r',
            data={
                'round': self._round * torch.ones([click_pair.shape[0]])
            }
        )
        users = like[:, 0]
        articles = like[:, 1]
        self._graph.add_edges(
            u=users,
            v=articles,
            etype='like',
            data={
                'round': self._round * torch.ones([like.shape[0]])
            }
        )
        self._graph.add_edges(
            u=articles,
            v=users,
            etype='like_r',
            data={
                'round': self._round * torch.ones([like.shape[0]])
            }
        )
    
    def update_pool(self, new_article_latent, new_article_quality):
        num_create = self._config['num_create']
        num_new_article = new_article_latent.shape[0]
        # embedding_dim = self._graph.nodes['article'].data['embedding'].shape[1]
        cid = torch.cat([torch.Tensor(list(range(self._num_node['creator']))).unsqueeze(1) for i in range(num_create)], dim=1).reshape(-1).type(torch.int32)
        iid = torch.Tensor(list(range(self._num_node['article'], self._num_node['article'] + num_new_article))).type(torch.int32)
        # add nodes
        self._graph.add_nodes(num=num_new_article, ntype='article', data={
            'quality': new_article_quality,
            'latent': new_article_latent,
            # 'embedding': torch.zeros([num_new_article, embedding_dim])
        })
        # add edges
        self._graph.add_edges(
            u=cid,
            v=iid,
            etype='create',
            data={
                'round': self._round * torch.ones([num_new_article])
            }
        )
        self._graph.add_edges(
            u=iid,
            v=cid,
            etype='create_r',
            data={
                'round': self._round * torch.ones([num_new_article])
            }
        )
        # update article num
        self._num_node['article'] += num_new_article

    def update_interuption(self, interruptions):
        pass

    def update_round(self):
        self._round += 1


