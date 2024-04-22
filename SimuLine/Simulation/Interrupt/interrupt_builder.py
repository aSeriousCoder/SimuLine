from SimuLine.Simulation.Interrupt.basic_interrupt import BasicInterrupt

class InterruptBuilder:
    def __init__(self, config, name='SimuLine-InterruptBuilder'):
        self._name = name
        self._config = config

    def build(self, metric):
        return BasicInterrupt(self._config, metric)