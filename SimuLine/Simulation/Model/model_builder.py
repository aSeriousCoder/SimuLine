from logging import getLogger
from SimuLine.Simulation.Model.data_transformer import DataTransformer
from recbole.utils import get_model, get_trainer


class ModelBuilder:
    def __init__(self, simulation_config, recbole_config, name='SimuLine-ModelBuilder'):
        self._name = name
        self._simulation_config = simulation_config
        self._recbole_config = recbole_config   
        self._logger = getLogger()

    def build(self, raw_data):
        # Data
        data_transformer = DataTransformer(self._recbole_config)
        if 'pos-neg' in self._recbole_config['version']:
            data_transformer.transform_explicit_dataset(raw_data)
        else:
            data_transformer.transform_implicit_dataset(raw_data)
        train_data, valid_data, test_data, dataset = data_transformer.build()
        # Model
        model = get_model(self._recbole_config['model'])(self._recbole_config, dataset).to(self._recbole_config['device'])
        self._logger.info(model)  # logging string controlled by recbole
        # Trainer
        trainer = get_trainer(self._recbole_config['MODEL_TYPE'], self._recbole_config['model'])(self._recbole_config, model)
        return trainer, model, train_data, valid_data, test_data, dataset


