from SimuLine.Simulation.Config.lists import MODEL_LIST_NONE_NEG_SAMPLING
from recbole.config import Config


def example_hook(simulation_config, recbole_config, round):
    if round == 2:
        simulation_config['num_from_hot'] = 10
        simulation_config['num_from_promote'] = 30
        recbole_config['epoch'] = 5
        recbole_config['eval_step'] = 5


ROUND_ALGORITHM_LIST = ['BPR', 'MultiVAE', 'NeuMF', 'NAIS', 'LightGCN', 'SpectralCF']

def round_algorithm_hook(simulation_config, round):
    simulation_config['model'] = ROUND_ALGORITHM_LIST[(round-1) % len(ROUND_ALGORITHM_LIST)]
    if simulation_config['model'] in MODEL_LIST_NONE_NEG_SAMPLING:
        simulation_config['neg_sampling'] = None
    else:
        simulation_config['neg_sampling'] = {'uniform': 9}
    recbole_config = Config(config_dict=simulation_config)
    return simulation_config, recbole_config



    