
from logging import getLogger
from recbole.utils import set_color

class BasicInterrupt:
    def __init__(self, config, metric, name='SimuLine-BasicInterrupt'):
        self._name = name
        self._config = config
        self._metric = metric
        self._logger = getLogger()
    
    def log_info(self, info):
        info_str = f'[{self._name}] {info}'
        self._logger.info(set_color(info_str, 'blue'))
    
    def make(self, graph_dataset):
        self.log_info('Pass')