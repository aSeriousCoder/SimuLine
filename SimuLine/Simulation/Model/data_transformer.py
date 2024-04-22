import torch
import numpy as np
import pandas as pd
from logging import getLogger
from recbole.data import create_dataset, data_preparation
from recbole.utils import set_color
from random import choice

class DataTransformer:
    def __init__(self, recbole_config, name='SimuLine-DataTransformer'):
        self._name = name
        self._recbole_config = recbole_config
        self._logger = getLogger()
        self._map = {
            'like': 0,
            'click': 1,
            'exposure': 2,
        }
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))
    
    def transform_explicit_dataset(self, raw_data):
        '''
        We need do neg_sampling here, as the "label-system" doesn't really work ...
        '''
        self.log_info('Transforming Interaction Dataset')
        # process raw_data
        positive_interaction = raw_data[self._map[self._recbole_config['positive_inter_type']]][:, :2]
        negative_interaction = raw_data[self._map[self._recbole_config['negative_inter_type']]][:, :2]
        exposure_interaction = raw_data[2][:, :2]
        user_wz_pos = positive_interaction[:, 0].unique().tolist()
        raw_data = {uid:{'pos':[],'neg':[],'exposure':[]} for uid in user_wz_pos}
        for u, i in positive_interaction.tolist():
            raw_data[u]['pos'].append(i)
        for u, i in negative_interaction.tolist():
            if u in user_wz_pos and i not in raw_data[u]['pos']:
                raw_data[u]['neg'].append(i)
        for u, i in exposure_interaction.tolist():
            if u in user_wz_pos and i not in raw_data[u]['pos'] and i not in raw_data[u]['neg']:
                raw_data[u]['exposure'].append(i)
        paired_data = []
        for u in user_wz_pos:
            assert len(raw_data[u]['pos']) > 0
            if len(raw_data[u]['neg']) > 0: # sample from neg
                for pos_i in raw_data[u]['pos']:
                    for neg_i in raw_data[u]['neg']:
                        paired_data.append([u, pos_i, neg_i])
            else: # sample from exposure
                for pos_i in raw_data[u]['pos']:
                    for j in range(9):
                        paired_data.append([u, pos_i, choice(raw_data[u]['exposure'])])
        assert len(torch.Tensor(paired_data).int()[:, 0].unique()) == len(user_wz_pos)
        paired_data = [[str(pairdata[0]), str(pairdata[1]), str(pairdata[2])] for pairdata in paired_data]
        # positive_interaction[:, 0].unique().shape[0]
        # positive_interaction[:, 1].unique().shape[0]
        # # ---
        # # old version, error of missing users
        # # clean positive from negative, then join
        # pos_df = pd.DataFrame(positive_interaction.numpy().astype(np.int32))
        # neg_df = pd.DataFrame(negative_interaction.numpy().astype(np.int32))
        # neg_df = pd.concat([neg_df, pos_df, pos_df]).drop_duplicates(keep=False)
        # pos_df.columns = ['user_id', 'item_id']
        # neg_df.columns = ['user_id', 'neg_item_id']
        # data = pd.merge(pos_df, neg_df, how = 'inner', on='user_id').values.astype(np.int32).astype(np.str).tolist()
        # # ---
        # Write Data
        interaction_columns = ['user_id:token', 'item_id:token', 'neg_item_id:token']
        interaction_columns_str  = '\t'.join(interaction_columns)
        interaction_data_str = '\n'.join(['\t'.join(record) for record in paired_data])
        interaction_str = f"{interaction_columns_str}\n{interaction_data_str}"
        interaction_path = f"{self._recbole_config['data_path']}/{self._recbole_config['dataset']}.inter"
        with open(interaction_path, 'w') as f:
            f.write(interaction_str)
    
    def transform_implicit_dataset(self, raw_data):
        self.log_info('Transforming Interaction Dataset')
        # process raw_data
        positive_interaction = raw_data[self._map[self._recbole_config['positive_inter_type']]][:, :2]
        interaction_data = positive_interaction.numpy().astype(np.str).tolist()
        # Write Data
        interaction_columns = ['user_id:token', 'item_id:token']
        interaction_columns_str  = '\t'.join(interaction_columns)
        interaction_data_str = '\n'.join(['\t'.join(record) for record in interaction_data])
        interaction_str = f"{interaction_columns_str}\n{interaction_data_str}"
        interaction_path = f"{self._recbole_config['data_path']}/{self._recbole_config['dataset']}.inter"
        with open(interaction_path, 'w') as f:
            f.write(interaction_str)
    
    def build(self):
        self.log_info('Build')
        dataset = create_dataset(self._recbole_config)
        self._logger.info(dataset)  # logging string controlled by recbole
        train_data, valid_data, test_data = data_preparation(self._recbole_config, dataset)
        return train_data, valid_data, test_data, dataset

