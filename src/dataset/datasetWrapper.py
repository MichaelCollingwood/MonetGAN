class DatasetWrapper:
    def __init__(self, image_folder):
        from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize

        self.image_folder = image_folder
        self.image_size = 64

        self.transforms = [
            Resize(self.image_size),
            CenterCrop(self.image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    def dataloader(self, config: dict):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset=self.dataset(),
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["workers"]
        )

    def dataset(self):
        import torchvision.datasets as dset

        return dset.ImageFolder(
            root=self.image_folder,
            transform=self.transform()
        )

    def transform(self):
        from torchvision.transforms import Compose

        return Compose(self.transforms)
