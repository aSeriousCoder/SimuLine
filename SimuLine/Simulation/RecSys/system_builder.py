from SimuLine.Simulation.RecSys.basic_system import BasicSystem

class SystemBuilder:
    def __init__(self, config, name='SimuLine-SystemBuilder'):
        self._name = name
        self._config = config

    def build(self, metric):
        return BasicSystem(self._config, metric)