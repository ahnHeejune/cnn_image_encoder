import numpy as np
import torch

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class CosineSimilarityMetric:

    def __init__(self):
        self.total_cos_sim = 0
        self.count = 0

    def reset(self):
        self.total_cos_sim = 0
        self.count = 0

    def __call__(self, outputs, target, loss_outputs):
    
        # now triplet only 
        # outputs가 (z_a, z_p, z_n) 튜플이라고 가정
        z_a, z_p = outputs[0], outputs[1]
        
        # 내적 계산
        cos_sim = torch.sum(z_a * z_p, dim=1)
        
        self.total_cos_sim += cos_sim.sum().item()
        self.count += z_a.size(0)

    def value(self):
        return self.total_cos_sim / self.count if self.count > 0 else 0


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
