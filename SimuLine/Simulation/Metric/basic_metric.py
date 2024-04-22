import torch
import dgl
import numpy as np
import pandas as pd
from torch.nn.functional import normalize

from logging import getLogger
from recbole.utils import set_color

from SimuLine.Simulation.Graph.dgl_function import edges_this_round, edges_last_round
from SimuLine.Simulation.Graph.dgl_function import msg_user_article_similarity, reduce_user_article_similarity
from SimuLine.Simulation.Graph.dgl_function import msg_cross_round_article_shifting, reduce_cross_round_article_shifting
from SimuLine.Simulation.Graph.dgl_function import msg_inlist_article_similarity, reduce_inlist_article_similarity

EPSILONE = 1e-8

class BasicMetric:
    def __init__(self, config, name='SimuLine-BasicMetric'):
        self._name = name
        self._config = config
        self._exp = self._config['experiment']
        self._var = self._config['var']
        self._run = self._config['run']
        self._home = '.'
        self._logger = getLogger()
        self._static_record = {
            'user_threshold': None,
            'user_like_quality_weight': None,
            'user_like_match_weight': None,
            'user_concentration': None,
            'creator_concentration': None,
        }
        self._user_record = {
            'latent': [],
            'like': [],  # num like
            'liked_quality_contribution': [],
            'liked_match_contribution': [],
        }
        self._creator_record = {
            'latent': [],
            'exposure': [],  # num exposure
            'click': [],  # num click
            'like': [],  # num like
        }
        self._article_record = {  
            # latent is static, could be obtain from DGL_Graph directly
            # only active articles
            'latent': None,
            'exposure': [],  # acc exposure
            'click': [],  # acc click
            'like': [],  # acc like
            'quality': [],  # quality
        }
        self._recsys_record = {
            'mrr@5': [],
            'num_registed_user': [],
            'num_registed_article': [],
        }
        self._output = {

            # 1. Metric
            # 1.1 Interaction
            'Metric-Inter-User-Like-Mean': [],
            'Metric-Inter-User-Like-Gini': [],
            'Metric-Inter-User-Like-Quality_Contribution': [],
            'Metric-Inter-User-Like-Match_Contribution': [],

            'Metric-Inter-Creator-Exposure-Gini': [],
            'Metric-Inter-Creator-Click-Gini': [],
            'Metric-Inter-Creator-Like-Mean': [],
            'Metric-Inter-Creator-Like-Gini': [],

            'Metric-Inter-Article-Exposure-Gini': [],
            'Metric-Inter-Article-Click-Gini': [],
            'Metric-Inter-Article-Like-Mean': [],
            'Metric-Inter-Article-Like-Gini': [],
            
            # 1.2 Quality
            'Metric-Quality-Create-Weighted': [],
            'Metric-Quality-Create-Gini': [],
            'Metric-Quality-Exposure-Weighted': [],
            'Metric-Quality-Click-Weighted': [],
            'Metric-Quality-Like-Weighted': [],
            'Metric-Quality-Exposure-Corrcoef': [],
            'Metric-Quality-Click-Corrcoef': [],
            'Metric-Quality-Like-Corrcoef': [],

            # 1.3 Homogenization
            'Metric-Homogenization-User-Exposure': [],
            'Metric-Homogenization-User-Click': [],
            'Metric-Homogenization-User-Like': [],

            # 1.4 Recsys Infomation
            'Metric-RecSys-MRR@5': [],
            'Metric-RecSys-User-Registed': [],
            'Metric-RecSys-Article-Registed': [],

            # 2. Latent
            'Latent-Cross_Round_Article_Shifting-Exposure': [],
            'Latent-Cross_Round_Article_Shifting-Click': [],
            'Latent-Cross_Round_Article_Shifting-Like': [],
            'Latent-In_List_Article_Similarity-Exposure': [],
            'Latent-In_List_Article_Similarity-Click': [],
            'Latent-In_List_Article_Similarity-Like': [],
            'Latent-User_Article_Similarity-Exposure': [],
            'Latent-User_Article_Similarity-Click': [],
            'Latent-User_Article_Similarity-Like': [],
            'Latent-User_Interest_Shifting': [],
        }
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))

    def eval(self, graph_dataset):

        self.log_info('Record raw data')
        # ---  Prepare  ---
        total_num_create_each_round = self._config['num_creator'] * self._config['num_create']
        earliest_active_round = max(0, graph_dataset._round - self._config['n_round'])
        item_index_padding = earliest_active_round * total_num_create_each_round  # articles with index before this are unactivited
        num_active_item = graph_dataset.num_node['article'] - item_index_padding  # needn't to exclude newly created articles
        sg_recent_N_round_article = dgl.node_subgraph(graph_dataset.graph, {
            'user': torch.Tensor(range(graph_dataset.num_node['user'])).int(),  # all user
            'article': torch.Tensor(range(item_index_padding, item_index_padding + num_active_item)).int(),  # active articles
            'creator': torch.Tensor(range(graph_dataset.num_node['creator'])).int(),  # all creator
        })
        # ---  User  ---
        # latent
        self.user_record['latent'].append(sg_recent_N_round_article.ndata['latent']['user'])
        # like
        self.user_record['like'].append(sg_recent_N_round_article.in_degrees(etype='like_r'))
        # ---  Creator  ---
        # latent
        self.creator_record['latent'].append(sg_recent_N_round_article.ndata['latent']['creator'])
        # exposure
        creator_item_matrix = sg_recent_N_round_article.adj(etype = 'create').to_dense().type(torch.int32)
        item_degree_exposure = sg_recent_N_round_article.in_degrees(etype='exposure')
        creator_item_degree_matrix_exposure = creator_item_matrix * item_degree_exposure.unsqueeze(0).expand(creator_item_matrix.shape)
        creator_degree_exposure = creator_item_degree_matrix_exposure.sum(dim=1)
        self.creator_record['exposure'].append(creator_degree_exposure)
        # click
        item_degree_click = sg_recent_N_round_article.in_degrees(etype='click')
        creator_item_degree_matrix_click = creator_item_matrix * item_degree_click.unsqueeze(0).expand(creator_item_matrix.shape)
        creator_degree_click = creator_item_degree_matrix_click.sum(dim=1)
        self.creator_record['click'].append(creator_degree_click)
        # like
        item_degree_like = sg_recent_N_round_article.in_degrees(etype='like')
        creator_item_degree_matrix_like = creator_item_matrix * item_degree_like.unsqueeze(0).expand(creator_item_matrix.shape)
        creator_degree_like = creator_item_degree_matrix_like.sum(dim=1)
        self.creator_record['like'].append(creator_degree_like)
        # ---  Article  ---
        # latent
        self.article_record['latent']= graph_dataset.graph.ndata['latent']['article']
        # exposure
        self.article_record['exposure'].append(item_degree_exposure)
        # click
        self.article_record['click'].append(item_degree_click)
        # like
        self.article_record['like'].append(item_degree_like)
        # quality
        self.article_record['quality'].append(sg_recent_N_round_article.ndata['quality']['article'])


        self.log_info('Eval')
        # 1. Metric
        # 1.1 Interaction
        self.output['Metric-Inter-User-Like-Mean'].append(
            float(self.user_record['like'][-1].float().mean())
        )
        self.output['Metric-Inter-User-Like-Gini'].append(
            self.gini(self.user_record['like'][-1].float())
        )
        if graph_dataset.round == 0:
            self.output['Metric-Inter-User-Like-Quality_Contribution'].append(0.0)
            self.output['Metric-Inter-User-Like-Match_Contribution'].append(0.0)
        else:
            self.output['Metric-Inter-User-Like-Quality_Contribution'].append(
                float(self.user_record['liked_quality_contribution'][-1].float().mean())
            )
            self.output['Metric-Inter-User-Like-Match_Contribution'].append(
                float(self.user_record['liked_match_contribution'][-1].float().mean())
            )
        self.output['Metric-Inter-Creator-Exposure-Gini'].append(
            self.gini(self.creator_record['exposure'][-1].float())
        )
        self.output['Metric-Inter-Creator-Click-Gini'].append(
            self.gini(self.creator_record['click'][-1].float())
        )
        self.output['Metric-Inter-Creator-Like-Mean'].append(
            float(self.creator_record['like'][-1].float().mean())
        )
        self.output['Metric-Inter-Creator-Like-Gini'].append(
            self.gini(self.creator_record['like'][-1].float())
        )
        self.output['Metric-Inter-Article-Exposure-Gini'].append(
            self.gini(self.article_record['exposure'][-1].float())
        )
        self.output['Metric-Inter-Article-Click-Gini'].append(
            self.gini(self.article_record['click'][-1].float())
        )
        self.output['Metric-Inter-Article-Like-Mean'].append(
            float(self.article_record['like'][-1].float().mean())
        )
        self.output['Metric-Inter-Article-Like-Gini'].append(
            self.gini(self.article_record['like'][-1].float())
        )

        # 1.2 Quality
        self.output['Metric-Quality-Create-Weighted'].append(
            float(self.article_record['quality'][-1].float().mean())
        )
        if graph_dataset.round == 0:
            self.output['Metric-Quality-Create-Gini'].append(0.0)
        else:
            self.output['Metric-Quality-Create-Gini'].append(
                self.gini(self.article_record['quality'][-1].float())
            )
        self.output['Metric-Quality-Exposure-Weighted'].append(
            float((self.article_record['quality'][-1] * self.article_record['exposure'][-1] / self.article_record['exposure'][-1].sum()).sum())
        )
        self.output['Metric-Quality-Click-Weighted'].append(
            float((self.article_record['quality'][-1] * self.article_record['click'][-1] / self.article_record['click'][-1].sum()).sum())
        )
        self.output['Metric-Quality-Like-Weighted'].append(
            float((self.article_record['quality'][-1] * self.article_record['like'][-1] / self.article_record['like'][-1].sum()).sum())
        )
        if graph_dataset.round == 0:
            self.output['Metric-Quality-Exposure-Corrcoef'].append(0.0)
            self.output['Metric-Quality-Click-Corrcoef'].append(0.0)
            self.output['Metric-Quality-Like-Corrcoef'].append(0.0)
            # 1.4 Recsys Infomation
            self.output['Metric-RecSys-MRR@5'].append(0.0)
            self.output['Metric-RecSys-User-Registed'].append(0.0)
            self.output['Metric-RecSys-Article-Registed'].append(0.0)
        else:
            self.output['Metric-Quality-Exposure-Corrcoef'].append(
                np.corrcoef(self.article_record['quality'][-1], self.article_record['exposure'][-1])[0,1]
            )
            self.output['Metric-Quality-Click-Corrcoef'].append(
                np.corrcoef(self.article_record['quality'][-1], self.article_record['click'][-1])[0,1]
            )
            self.output['Metric-Quality-Like-Corrcoef'].append(
                np.corrcoef(self.article_record['quality'][-1], self.article_record['like'][-1])[0,1]
            )
            # 1.4 Recsys Infomation
            self.output['Metric-RecSys-MRR@5'].append(self._recsys_record['mrr@5'][-1])
            self.output['Metric-RecSys-User-Registed'].append(self._recsys_record['num_registed_user'][-1])
            self.output['Metric-RecSys-Article-Registed'].append(self._recsys_record['num_registed_article'][-1])

        # 1.3 Homogenization
        sample_index = np.random.choice(list(range(self._config['num_user'])), int(self._config['num_user']/10), replace=False)
        creator_item_matrix_exposure = sg_recent_N_round_article.adj(etype = 'exposure').to_dense().type(torch.int32).numpy()  # this is a quite big matrix
        self.output['Metric-Homogenization-User-Exposure'].append(
            self.jaccard(creator_item_matrix_exposure[sample_index]).mean()
        )
        creator_item_matrix_click = sg_recent_N_round_article.adj(etype = 'click').to_dense().type(torch.int32).numpy()  # this is a quite big matrix
        self.output['Metric-Homogenization-User-Click'].append(
            self.jaccard(creator_item_matrix_click[sample_index]).mean()
        )
        creator_item_matrix_like = sg_recent_N_round_article.adj(etype = 'like').to_dense().type(torch.int32).numpy()  # this is a quite big matrix
        self.output['Metric-Homogenization-User-Like'].append(
            self.jaccard(creator_item_matrix_like[sample_index]).mean()
        )

        # 2. Latent
        graph_dataset.sync_round()
        like_r_eid_this_round = sg_recent_N_round_article.filter_edges(edges_this_round, etype='like_r')
        click_r_eid_this_round = sg_recent_N_round_article.filter_edges(edges_this_round, etype='click_r')
        exposure_r_eid_this_round = sg_recent_N_round_article.filter_edges(edges_this_round, etype='exposure_r')
        sg_this_round = dgl.edge_subgraph(sg_recent_N_round_article, {
            'like_r': like_r_eid_this_round.int(),
            'click_r': click_r_eid_this_round.int(),
            'exposure_r': exposure_r_eid_this_round.int(),
        })  # missing some nodes in this subgraph is ok, as they contribute 0 to the final value
        sg_this_round.update_all(msg_user_article_similarity, reduce_user_article_similarity, etype=('article', 'exposure_r', 'user'))
        self.output['Latent-User_Article_Similarity-Exposure'].append(
            float((sg_this_round.ndata['user_article_mean_similarity']['user'] * sg_this_round.in_degrees(etype='exposure_r') / sg_this_round.in_degrees(etype='exposure_r').sum()).sum())
        )
        sg_this_round.update_all(msg_user_article_similarity, reduce_user_article_similarity, etype=('article', 'click_r', 'user'))
        self.output['Latent-User_Article_Similarity-Click'].append(
            float((sg_this_round.ndata['user_article_mean_similarity']['user'] * sg_this_round.in_degrees(etype='click_r') / sg_this_round.in_degrees(etype='click_r').sum()).sum())
        )
        sg_this_round.update_all(msg_user_article_similarity, reduce_user_article_similarity, etype=('article', 'like_r', 'user'))
        self.output['Latent-User_Article_Similarity-Like'].append(
            float((sg_this_round.ndata['user_article_mean_similarity']['user'] * sg_this_round.in_degrees(etype='like_r') / sg_this_round.in_degrees(etype='like_r').sum()).sum())
        )

        sg_this_round.update_all(msg_inlist_article_similarity, reduce_inlist_article_similarity, etype=('article', 'exposure_r', 'user'))
        self.output['Latent-In_List_Article_Similarity-Exposure'].append(
            float(sg_this_round.ndata['inlist_article_similarity']['user'].mean())
        )
        sg_this_round.update_all(msg_inlist_article_similarity, reduce_inlist_article_similarity, etype=('article', 'click_r', 'user'))
        self.output['Latent-In_List_Article_Similarity-Click'].append(
            float(sg_this_round.ndata['inlist_article_similarity']['user'].mean())
        )
        sg_this_round.update_all(msg_inlist_article_similarity, reduce_inlist_article_similarity, etype=('article', 'like_r', 'user'))
        self.output['Latent-In_List_Article_Similarity-Like'].append(
            float((sg_this_round.ndata['inlist_article_similarity']['user'] * sg_this_round.ndata['inlist_article_similarity_weight']['user'] / sg_this_round.ndata['inlist_article_similarity_weight']['user'].sum()).sum())
        )

        if graph_dataset.round == 0:
            self.output['Latent-Cross_Round_Article_Shifting-Exposure'].append(0.0)
            self.output['Latent-Cross_Round_Article_Shifting-Click'].append(0.0)
            self.output['Latent-Cross_Round_Article_Shifting-Like'].append(0.0)
            self.output['Latent-User_Interest_Shifting'].append(0.0)
        else:
            graph_dataset.sync_round()
            sg_recent_N_round_article.update_all(msg_cross_round_article_shifting, reduce_cross_round_article_shifting, etype=('article', 'exposure_r', 'user'))
            self.output['Latent-Cross_Round_Article_Shifting-Exposure'].append(
                float(sg_recent_N_round_article.ndata['cross_round_article_shifting']['user'].mean())
            )
            sg_recent_N_round_article.update_all(msg_cross_round_article_shifting, reduce_cross_round_article_shifting, etype=('article', 'click_r', 'user'))
            self.output['Latent-Cross_Round_Article_Shifting-Click'].append(
                float(sg_recent_N_round_article.ndata['cross_round_article_shifting']['user'].mean())
            )
            sg_recent_N_round_article.update_all(msg_cross_round_article_shifting, reduce_cross_round_article_shifting, etype=('article', 'like_r', 'user'))
            self.output['Latent-Cross_Round_Article_Shifting-Like'].append(
                float((sg_recent_N_round_article.ndata['cross_round_article_shifting']['user'] * sg_recent_N_round_article.ndata['cross_round_article_shifting_weight']['user'] / sg_recent_N_round_article.ndata['cross_round_article_shifting_weight']['user'].sum()).sum())
            )
            self.output['Latent-User_Interest_Shifting'].append(
                float(torch.norm((self._user_record['latent'][-1] - self._user_record['latent'][-2]), p=2, dim=1).mean())
            )

    def gini(self, x):
        cum_x = np.cumsum(sorted(np.append(x, 0)))
        sum_x = cum_x[-1]
        xarray = np.array(range(0, len(cum_x))) / np.float(len(cum_x) - 1)
        yarray = cum_x / sum_x
        B = np.trapz(yarray, x=xarray)
        A = 0.5 - B
        G = A / (A + B)
        return G
    
    def jaccard(self, X):
        # assert X.max() == 1
        intersect = np.matmul(X, X.T)
        union = X.shape[1] - np.matmul((1-X), (1-X).T)
        jaccard = intersect / (union + EPSILONE)
        return jaccard
 
    def write(self):
        results_df = pd.DataFrame.from_dict(self.output, orient='index')
        results_df.to_csv('{}/Out/Result/{}/{}/{}_output.csv'.format(self._home, self._exp, self._var, self._run))
        torch.save(self.static_record, '{}/Out/Result/{}/{}/{}_static_record.pth'.format(self._home, self._exp, self._var, self._run))
        torch.save(self.user_record, '{}/Out/Result/{}/{}/{}_user_record.pth'.format(self._home, self._exp, self._var, self._run))
        torch.save(self.creator_record, '{}/Out/Result/{}/{}/{}_creator_record.pth'.format(self._home, self._exp, self._var, self._run))
        torch.save(self.article_record, '{}/Out/Result/{}/{}/{}_article_record.pth'.format(self._home, self._exp, self._var, self._run))
        torch.save(self.recsys_record, '{}/Out/Result/{}/{}/{}_recsys_record.pth'.format(self._home, self._exp, self._var, self._run))

    @property
    def user_record(self):
        return self._user_record
    
    @property
    def creator_record(self):
        return self._creator_record
    
    @property
    def article_record(self):
        return self._article_record
    
    @property
    def recsys_record(self):
        return self._recsys_record

    @property
    def static_record(self):
        return self._static_record
    
    @property
    def output(self):
        return self._output

