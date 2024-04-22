from SimuLine.Simulation.Metric.basic_metric import BasicMetric

class MetricBuilder:
    def __init__(self, config, name='SimuLine-MetricBuilder'):
        self._name = name
        self._config = config

    def build(self):
        return BasicMetric(self._config)