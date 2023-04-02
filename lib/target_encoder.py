import numpy as np
from collections import defaultdict


class TargetEncoder:
    def fit_transform(self, data, targets):
        mean_target = np.mean(targets)
        self._counts = defaultdict(lambda: 1.0)
        self._targets = defaultdict(lambda: mean_target)

        result = []
        for item, target in zip(data, targets):
            result.append(self._targets[item] / self._counts[item])
            self._targets[item] += target
            self._counts[item] += 1

        return np.float32(result)

    def transform(self, data):
        return np.float32([self._targets[item] / self._counts[item] for item in data])
