from torch import nn


def initWeights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ModelWrapper:
    def __init__(self, params):
        self.params = params
        self.net = None
        self.epoch_offset = 0
        pass

    def loadWeights(self, weights_file=None):
        pass

    def parameters(self):
        return self.net.parameters()
