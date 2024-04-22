from SimuLine.Simulation.User.basic_uam import BasicUAM

class UAMBuilder:
    def __init__(self, config, name='SimuLine-UAMBuilder'):
        self._name = name
        self._config = config

    def build(self, metric):
        return BasicUAM(self._config, metric)
