from generator import Generator
from plotter import Plotter


class Monet:
    def __init__(self) -> None:
        workers: int = 15  # Number of workers for dataloader
        batch_size: int = 512  # Batch size during training
        image_size: int = 64  # Spatial size of training images
        nc: int = 3  # Number of channels in the training images. For color images this is 3
        nz: int = 100  # Size of z latent vector (i.e. size of generator input)
        ngf: int = 64  # Size of feature maps in generator
        ndf: int = 64  # Size of feature maps in discriminator
        num_epochs: int = 1000  # Number of training epochs
        lr: float = 0.001  # Learning rate for optimizers
        beta1: float = 0.5  # Beta1 hyperparam for Adam optimizers
        gpu_count: int = 15  # Number of GPUs available. Use 0 for CPU mode.

        self.generator = Generator(gpu_count, nz, ngf, nc)
        self.plotter = Plotter()


    def train(self):
        pass

    def paint(self):
        """
        Predicts painted images.

        Parameters:
            random_seed: Int    Random seed used for predicting an image. Keep the same for repeatability.

        Returns:
            image: numpy.array  Redicted image.
        """
        pass
