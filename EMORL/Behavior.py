import numpy as np


class Behavior:

    def __init__(self, categories):
        self.stats = {c: np.nan for c in categories}

    def get_stats(self):
        return np.array([v for v in self.stats.values()])


