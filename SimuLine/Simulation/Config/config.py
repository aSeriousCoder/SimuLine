from copy import deepcopy


# The Basic Config
BASE_CONFIG = {

    # PART 1: RecBole Config

    # General Configs
    'description': 'base',
    'experiment': 'test',
    'var': 'var1',
    'run': 'run1',
    'version': 'test_var1_run1',
    'model': 'BPR',
    'dataset': 'SimuLine',
    'data_path': './SimuLine/Simulation/Data/tmp',
    'checkpoint_dir': './log/ckpt',

    # Training Configs
    'epochs': 100,
    'train_batch_size': 1024,
    'learner': 'adam',
    'learning_rate': 1e-3,
    'neg_sampling': {'uniform': 9},  # range is ['uniform', 'popularity']
    'eval_step': 10,
    'stopping_step': 3,

    # Evaluation Configs
    'eval_args': {
        'split': {'RS': [0.8,0.1,0.1]},
        'group_by': 'user',
        'order': 'RO',
        'mode': 'uni9',  # Range in [labeled, full, unixxx, popxxx]
    },
    'repeatable': True,
    'eval_batch_size': 2048,
    'metrics': [
        'Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC',  # accuracy metrics
        'ItemCoverage', 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage',  # non-accuracy metrics
    ],
    'topk': 5,
    'valid_metric': 'MRR@5',  # @topk

    # Customize Configs
    'positive_inter_type': 'like',  # like / click

    # --------------------------------------------------------------------
    
    # PART 2: Simulation Config
    
    # Simulation Config
    'num_round': 200,
    'num_user': 10000,
    'num_creator': 1000,
    'n_round': 5,  # num of round article being active

    # Graph Config
    'init_quality_mean': 0,  # all set to "0"
    'init_quality_var': 0,
    'best_stable_quality': 10.82,  # math.log(num_create*num_user+1)
    
    'init_threshold_mean': 3e-1,
    'init_threshold_var': 1e-1,
    'init_like_quality_weight_mean': 5e-1,
    'init_like_quality_weight_var': 1e-1,
    'init_user_concentration_mean': 5e-1,  # fix concentration once inited
    'init_user_concentration_var': 1e-1,
    'init_creator_concentration_mean': 5e-1,
    'init_creator_concentration_var': 1e-1,

    # RecSys Config
    'recommendation_list_length': 100,
    'num_from_match': 80,
    'num_from_cold_start': 20,
    'num_from_hot': 0,
    'num_from_promote': 0,  # also the num of content focus / author to be promoted
    'cold_start_inter_type': 'random',  # random / like / click
    'hot_inter_type': 'like',
    'promote_type': 'author',  # content / author
    'promote_round': 10,  # the promotion ends after n round, and renew the promote list

    # UAM & CAM Config
    'num_click': 10,
    'num_create': 5,
    'uam_delta': 1e-1,  # controling the evolution step
    'cam_delta': 1e-1,  # controling the evolution step
    'creator_target_inter_type': 'like',

}




# The Basic Labeled Config
BASE_LABELED_CONFIG = deepcopy(BASE_CONFIG)
BASE_LABELED_CONFIG['description'] = 'base_labeled'
BASE_LABELED_CONFIG['experiment'] = 'test'
BASE_LABELED_CONFIG['var'] = 'var2'
BASE_LABELED_CONFIG['run'] = 'run1'
BASE_LABELED_CONFIG['version'] = '{}_{}_{}'.format(BASE_LABELED_CONFIG['experiment'], BASE_LABELED_CONFIG['var'], BASE_LABELED_CONFIG['run'])
BASE_LABELED_CONFIG['positive_inter_type'] = 'like'  # like / click / exposure
BASE_LABELED_CONFIG['negative_inter_type'] = 'click'
BASE_LABELED_CONFIG['load_col'] = {'inter': ['user_id', 'item_id', 'neg_item_id']}
BASE_LABELED_CONFIG['neg_sampling'] = None  # we have done neg_sampling by ourselves
BASE_LABELED_CONFIG['alias_of_item_id'] = ['neg_item_id']  # List of fields’ names, which will be remapped into the same index system with ITEM_ID_FIELD

