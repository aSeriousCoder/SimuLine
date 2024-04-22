from SimuLine.Simulation.Graph.graph_builder import GraphBuilder
from SimuLine.Simulation.Model.model_builder import ModelBuilder
from SimuLine.Simulation.RecSys.system_builder import SystemBuilder
from SimuLine.Simulation.User.user_action_model_builder import UAMBuilder
from SimuLine.Simulation.Creator.creator_action_model_builder import CAMBuilder
from SimuLine.Simulation.Interrupt.interrupt_builder import InterruptBuilder
from SimuLine.Simulation.Metric.metric_builder import MetricBuilder
from SimuLine.Simulation.Config.config_hooks import round_algorithm_hook
from SimuLine.Simulation.Config.lists import MODEL_LIST_NONE_NEG_SAMPLING
from SimuLine.Simulation.Util.utils import dict_to_str, check_dir
from recbole.config import Config
from recbole.utils import init_logger, init_seed, set_color
from dgl import save_graphs
import torch

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=None, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# logging.basicConfig(filename='{}/Out/Log/{}/{}/{}.log'.format(HOME, EXP, VAR, RUN), level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
from logging import getLogger


def main(CONFIG):
    EXP = CONFIG['experiment']
    VAR = CONFIG['var']
    RUN = CONFIG['run']
    HOME = '.'
    check_dir('{}/Out/Log/{}'.format(HOME, EXP))
    check_dir('{}/Out/Log/{}/{}'.format(HOME, EXP, VAR))
    check_dir('{}/Out/Result/{}'.format(HOME, EXP))
    check_dir('{}/Out/Result/{}/{}'.format(HOME, EXP, VAR))
    check_dir('{}/Image/Simulation/{}'.format(HOME, EXP))
    check_dir('{}/Image/Simulation/{}/{}'.format(HOME, EXP, VAR))
    check_dir('{}/Image/Simulation/{}/{}/{}'.format(HOME, EXP, VAR, RUN))
    check_dir('{}/SimuLine/Simulation/Data/tmp/SimuLine_{}_{}_{}'.format(HOME, EXP, VAR, RUN))

    RECBOLE_CONFIG = Config(config_dict=CONFIG)
    if RECBOLE_CONFIG['model'] in MODEL_LIST_NONE_NEG_SAMPLING:
        RECBOLE_CONFIG['neg_sampling'] = None
        if 'labeled' in RECBOLE_CONFIG['description']:
            raise Exception('Please use UNLABELED versions')
    
    init_seed(RECBOLE_CONFIG['seed'], RECBOLE_CONFIG['reproducibility'])
    init_logger(RECBOLE_CONFIG)
    logger = getLogger()
    logger.info(dict_to_str(CONFIG, title='Simulation Hyper Parameters'))
    logger.info(RECBOLE_CONFIG)

    logger.info(set_color('[SimuLine-Pipeline] Build Modules', 'red'))
    metric = MetricBuilder(CONFIG).build()
    # metric as recorder
    graph_dataset = GraphBuilder(CONFIG).build(metric)
    system = SystemBuilder(CONFIG).build(metric)
    user_action_model = UAMBuilder(CONFIG).build(metric)
    creator_action_model = CAMBuilder(CONFIG).build(metric)
    interrupt = InterruptBuilder(CONFIG).build(metric)

    metric.eval(graph_dataset)
    graph_dataset.update_round()

    for rnd in range(1, CONFIG['num_round']+1):

        # A supported but not used feature 
        if CONFIG['version'] == 'RecSys-Config_algorithm_run6':
            cfg, rb_cfg = round_algorithm_hook(CONFIG, rnd)  # modify config while running
        else:
            cfg, rb_cfg = CONFIG, RECBOLE_CONFIG

        line = '-'*30
        logger.info(set_color(f'[SimuLine-Pipeline] {line}', 'red'))
        logger.info(set_color(f'[SimuLine-Pipeline] ROUND-{rnd} Begins', 'red'))

        # 1. Creator action: create new articles
        logger.info(set_color('[SimuLine-Pipeline] 1. Creator action: create new articles', 'red'))
        new_article_latent, new_article_quality = creator_action_model.action(graph_dataset)
        
        # 2. Add new articles in the graph_dataset
        logger.info(set_color('[SimuLine-Pipeline] 2. Add new articles in the graph_dataset', 'red'))
        graph_dataset.update_pool(new_article_latent, new_article_quality)
        
        # 3. Make interruptions
        # TODO implement interrupt
        logger.info(set_color('[SimuLine-Pipeline] 3. Make interruptions', 'red'))
        interruptions = interrupt.make(graph_dataset)
        
        # 4. Add interruptions in the graph_dataset
        logger.info(set_color('[SimuLine-Pipeline] 4. Add interruptions in the graph_dataset', 'red'))
        graph_dataset.update_interuption(interruptions)
        
        # 5. Extract raw data from dgl-graph
        logger.info(set_color('[SimuLine-Pipeline] 5. Extract raw data from dgl-graph', 'red'))
        raw_data = graph_dataset.interaction_records()
        logger.info(set_color('[SimuLine-Pipeline] Num of the users with like records: {}'.format(raw_data[0][:, 0].unique().shape[0]), 'white'))
        logger.info(set_color('[SimuLine-Pipeline] Num of the news with like records: {}'.format(raw_data[0][:, 1].unique().shape[0]), 'white'))
        logger.info(set_color('[SimuLine-Pipeline] Num of the news with click records: {}'.format(raw_data[1][:, 1].unique().shape[0]), 'white'))

        # 6. Build Dataset & Model & Trainer
        logger.info(set_color('[SimuLine-Pipeline] 6. Build Dataset & Model & Trainer', 'red'))
        model_builder = ModelBuilder(cfg, rb_cfg)
        trainer, model, train_data, valid_data, test_data, dataset = model_builder.build(raw_data)
        
        # 7. Train & Eval & Test
        logger.info(set_color('[SimuLine-Pipeline] 7. Train & Eval & Test', 'red'))
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=True, saved=True, show_progress=True
        )
        logger.info('{}: {}'.format(set_color('best valid score', 'blue'), best_valid_score))
        best_valid_result_str = '    '.join(['{} : {}'.format(k, v) for (k, v) in list(zip(list(best_valid_result.keys()), list(best_valid_result.values())))])
        logger.info('{}:\n{}'.format(set_color('best valid result', 'blue'), best_valid_result_str))
        # test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        # test_result_str = '    '.join(['{} : {}'.format(k, v) for (k, v) in list(zip(list(test_result.keys()), list(test_result.values())))])
        # logger.info('{}:\n{}'.format(set_color('test result', 'blue'), test_result_str))
        
        # 8. Export user & item embedding
        logger.info(set_color('[SimuLine-Pipeline] 8. Export user & item embedding', 'red'))
        user_id_mapping = dataset.field2id_token['user_id']
        item_id_mapping = dataset.field2id_token['item_id']
        user_id_field = rb_cfg['USER_ID_FIELD']
        item_id_field = rb_cfg['ITEM_ID_FIELD']

        metric.recsys_record['mrr@5'].append(best_valid_score)
        metric.recsys_record['num_registed_user'].append(user_id_mapping.shape[0])
        metric.recsys_record['num_registed_article'].append(item_id_mapping.shape[0])

        # 9. Mount model
        logger.info(set_color('[SimuLine-Pipeline] 9. Mount model', 'red'))
        system.mount_model(model, user_id_field, item_id_field, user_id_mapping, item_id_mapping, rb_cfg['device'], rb_cfg['eval_batch_size'])
        
        # 10. System make recommendation lists
        logger.info(set_color('[SimuLine-Pipeline] 10. System make recommendation lists', 'red'))
        exposure = system.recommend(graph_dataset)
        
        # 11. Update exposure in the graph_dataset
        logger.info(set_color('[SimuLine-Pipeline] 11. Update exposure in the graph_dataset', 'red'))
        graph_dataset.update_exposure(exposure)
       
        # 12. User action: click & interest_shift
        logger.info(set_color('[SimuLine-Pipeline] 12. User action: click & interest_shift', 'red'))
        click, like = user_action_model.action(graph_dataset, exposure)
        
        # 13. Update click & like in the graph_dataset
        logger.info(set_color('[SimuLine-Pipeline] 13. Update click & like in the graph_dataset', 'red'))
        graph_dataset.update_consumption(click, like)
        
        # 14. Evaluation of this round
        logger.info(set_color('[SimuLine-Pipeline] 14. Evaluation of this round', 'red'))
        metric.eval(graph_dataset)
        metric.write()
        
        # 15. Update round in the graph_dataset
        logger.info(set_color('[SimuLine-Pipeline] 15. Update round in the graph_dataset', 'red'))
        graph_dataset.update_round()
        graph_dataset.save()
        logger.info(set_color(f'[SimuLine-Pipeline] ROUND-{rnd} Finished', 'red'))

        # 16. Print result
        logger.info(dict_to_str(metric.output, title=f'Results @ Round-{rnd}'))

        # clean
        torch.cuda.empty_cache()

        # 17. Save DGL Graph
        save_graphs('{}/Out/Result/{}/{}/graph_{}.json'.format(HOME, EXP, VAR, rnd), [graph_dataset._graph])



