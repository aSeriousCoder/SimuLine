from SimuLine.Simulation.Creator.basic_cam import BasicCAM

class CAMBuilder:
    def __init__(self, config, name='SimuLine-CAMBuilder'):
        self._name = name
        self._config = config

    def build(self, metric):
        return BasicCAM(self._config, metric)