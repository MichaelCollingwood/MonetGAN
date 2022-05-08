from src.models.generator import Generator
from src.modelWrappers.modelWrapper import ModelWrapper


class GeneratorWrapper(ModelWrapper):
    def __init__(self, params):
        super().__init__(params)

        from torch import nn

        self.params = params
        self.epoch_offset = 0

        self.net = Generator(params)

        self.net = self.net.to(params["device"])
        if params["device"] == "cuda":
            self.net = nn.DataParallel(self.net, list(range(self.net.gpu_count)))

        self.optimizer = None
        self.losses = None

    def setupTraining(self, params):
        import torch.optim as optim

        self.optimizer = optim.Adam(self.net.parameters(), lr=params["lr"], betas=(params["beta"], 0.999))
        self.losses = []

    def __str__(self):
        print(self.net)

    def loadWeights(self, weights_file=None):
        import torch

        if weights_file is not None:
            ckpt = torch.load(weights_file)
            self.net.load_state_dict(ckpt['model_state_dict'])
            self.epoch_offset = ckpt['epoch']
        else:
            from src.modelWrappers.modelWrapper import initWeights
            self.net.apply(initWeights)

    def save(self, epoch, ckpt_dir):
        import torch

        epochs = epoch + self.epoch_offset
        epochs_str = ''.join(['0'] * (10 - epochs // 10)) + str(epochs)

        torch.save(
            {
                'model_state_dict': self.net.state_dict(),
                'epoch': epoch
            },
            f'{ckpt_dir}/generator_{epochs_str}.ckpt'
        )
