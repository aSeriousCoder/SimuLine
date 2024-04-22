
from SimuLine.Simulation.Graph.dgl_graph import SimuLineGraphDataset

class GraphBuilder:
    def __init__(self, config, name='SimuLine-GraphBuilder'):
        self._name = name
        self._config = config

    def build(self, metric):
        return SimuLineGraphDataset(self._config, metric, force_reload=True)