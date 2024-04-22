import torch
import dgl

from logging import getLogger
from recbole.utils import set_color

from SimuLine.Simulation.Graph.dgl_function import edges_in_prev_N_round
from SimuLine.Simulation.Creator.utils import quality_func

EPSILON = 1e-5


class BasicCAM:
    def __init__(self, config, metric, name='SimuLine-BasicCAM'):
        self._name = name
        self._config = config
        self._metric = metric
        self._logger = getLogger()
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))

    def action(self, graph_dataset):
        # 0. Part zero: Prepare
        self.log_info('Part zero: prepare')        
        total_num_create_each_round = self._config['num_creator'] * self._config['num_create']
        earliest_active_round = max(0, graph_dataset._round - self._config['n_round'])
        item_index_padding = earliest_active_round * total_num_create_each_round  # articles with index before this are unactivited
        num_active_item = graph_dataset.num_node['article'] - item_index_padding  # needn't to exclude newly created articles, as not created
        sg_recent_N_round_article = dgl.node_subgraph(graph_dataset.graph, {
            'user': torch.Tensor(range(graph_dataset.num_node['user'])).int(),  # all user
            'article': torch.Tensor(range(item_index_padding, item_index_padding + num_active_item)).int(),  # active articles
            'creator': torch.Tensor(range(graph_dataset.num_node['creator'])).int(),  # all creator
        })  # whether store_id makes no different
        item_latent = sg_recent_N_round_article.nodes['article'].data['latent']  # growing
        num_creator = graph_dataset.num_node['creator']
        creator_concentration = sg_recent_N_round_article.nodes['creator'].data['concentration']
        creator_target_inter_type = self._config['creator_target_inter_type']  # like

        # 1. Recent inter degrees
        self.log_info('Part one: collecting information')
        item_degree = sg_recent_N_round_article.in_degrees(etype=creator_target_inter_type)
        creator_item_matrix = sg_recent_N_round_article.adj(etype = 'create').to_dense().type(torch.int32)
        creator_item_degree_matrix = creator_item_matrix * item_degree.unsqueeze(0).expand(creator_item_matrix.shape)
        creator_concentration_ = creator_concentration.unsqueeze(1).expand(creator_item_degree_matrix.shape)
        selection_prob_matrix = (creator_concentration_ * creator_item_degree_matrix).softmax(dim=1)
        # selection_prob_matrix = ((-1 * (1 - creator_concentration_)) / (creator_item_degree_matrix.clip(min=0) + EPSILON)).exp().softmax(dim=1)
        # lower concentration -> flat distribution
         
        # 2. Select Topic
        self.log_info('Part two: sampling new articles')
        pick = torch.multinomial(selection_prob_matrix, num_samples=self._config['num_create'], replacement=True)  #  cor. to item_latent, no need to re-index
        newly_created_latent_mean = item_latent[pick.reshape(-1)]
        creator_concentration_ = creator_concentration.unsqueeze(1).unsqueeze(1).expand(num_creator, self._config['num_create'], item_latent.shape[-1]).reshape(num_creator * self._config['num_create'], item_latent.shape[-1])
        newly_created_latent = torch.normal(mean=newly_created_latent_mean, std=creator_concentration_ * self._config['cam_delta'])
        
        graph_dataset.graph.ndata['latent']['creator'] = newly_created_latent.reshape(num_creator, self._config['num_create'], item_latent.shape[-1]).mean(dim=1)

        # 3. Quality
        self.log_info('Part three: assign quality')
        creator_degree = creator_item_degree_matrix.sum(dim=1)
        newly_created_quality = quality_func(creator_degree.unsqueeze(1).expand(num_creator, self._config['num_create']).reshape(-1))

        return newly_created_latent, newly_created_quality


