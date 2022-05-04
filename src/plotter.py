from PlotNeuralNet.pycore.tikzeng import *


class Plotter:
    def __init__(self, model):
        self.architecture = self.getArchitecture(model)

    @staticmethod
    def getArchitecture(model):
        return model.shape()

    def plot(self, file):
        arch = [
            to_head('..'),
            to_cor(),
            to_begin()
        ] + [
            to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
            to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
            to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
            to_connection("pool1", "conv2"),
            to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
            to_SoftMax("soft1", 10, "(3,0,0)", "(pool1-east)", caption="SOFT"),
            to_connection("pool2", "soft1")
        ] + [
            to_end()
        ]

        to_generate(arch, file+'.tex')
