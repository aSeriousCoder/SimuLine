from SimuLine.Simulation.pipeline import main as simulation
from SimuLine.Simulation.Config.config import BASE_CONFIG as CONFIG
# from SimuLine.Simulation.Config.config import BASE_LABELED_CONFIG as CONFIG  # if negative_inter_type is not "-", use this instead of BASE_CONFIG
from copy import deepcopy
import pandas as pd


def main():
    experiment_settings = pd.read_excel('./Config/Batch.xlsx')
    for i in range(experiment_settings.shape[0]):
        config = deepcopy(CONFIG)
        for attr in experiment_settings.keys():
            attr_name, attr_type = attr.split('@')
            if experiment_settings[attr][i] == '-':  # pass "-" values which means no setting
                continue
            if attr_type == 'str':
                config[attr_name] = experiment_settings[attr][i]
            elif attr_type == 'int':
                config[attr_name] = int(experiment_settings[attr][i])
            elif attr_type == 'float':
                config[attr_name] = float(experiment_settings[attr][i])
            else:
                raise Exception('BAD CONFIGURATION FILE !!!')
        config['version'] = '{}_{}_{}'.format(config['experiment'], config['var'], config['run'])
        config['dataset'] = 'SimuLine_{}_{}_{}'.format(config['experiment'], config['var'], config['run'])
        simulation(config)


if __name__ == "__main__":
    main()

