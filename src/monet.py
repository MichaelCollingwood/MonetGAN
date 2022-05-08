import logging

from src.dataset.datasetWrapper import DatasetWrapper
from src.modelWrappers.discriminatorWrapper import DiscriminatorWrapper
from src.modelWrappers.generatorWrapper import GeneratorWrapper


class Monet:
    def __init__(
        self,
        pretrained_generator=None,
        pretrained_discriminator=None,
        training_dataset=None,
        **kwargs
    ) -> None:
        logging.info("Initialize Monet")

        import torch
        import torch.nn as nn

        self.hyper_params = {
            "workers": 0,  # Number of workers for dataloader
            "batch_size": 512,  # Batch size during training
            "image_size": 64,  # Spatial size of training images
            "shuffle": True,
            "nc": 3,  # Number of channels in the training images. For color images this is 3
            "nz": 100,  # Size of z latent vector (i.e. size of generator input)
            "ngf": 64,  # Size of feature maps in generator
            "ndf": 64,  # Size of feature maps in discriminator
            "lr": 0.001,  # Learning rate for optimizers
            "beta": 0.5,  # Beta1 hyper_param for Adam optimizers
            "gpu_count": 0  # Number of GPUs available. Use 0 for CPU mode.
        }
        self.hyper_params.update(kwargs)
        self.hyper_params["device"] = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.hyper_params["gpu_count"] > 0) else "cpu"
        )
        self.real_label = 1.
        self.fake_label = 0.
        self.criterion = nn.BCELoss()

        if training_dataset is not None:
            logging.info("Load dataloader")
            self.dataloader = DatasetWrapper(
                image_folder=training_dataset
            ).dataloader(
                config={k: v for k, v in self.hyper_params.items() if k in ["batch_size", "shuffle", "workers"]}
            )

        logging.info("Load models")
        self.G = GeneratorWrapper(
            params={k: v for k, v in self.hyper_params.items() if k in ["nz", "ngf", "nc", "device"]},
        )
        if training_dataset is not None:
            self.G.setupTraining(
                params={k: v for k, v in self.hyper_params.items() if k in ["lr", "beta"]}
            )
        self.G.loadWeights(pretrained_generator)

        self.D = DiscriminatorWrapper(
            params={k: v for k, v in self.hyper_params.items() if k in ["ndf", "nc", "device"]}
        )
        if training_dataset is not None:
            self.D.setupTraining(
                params={k: v for k, v in self.hyper_params.items() if k in ["lr", "beta"]}
            )
        self.D.loadWeights(pretrained_discriminator)

    def train(self, num_epochs, ckpt_dir, **kwargs):
        logging.info("Train model")

        import torch
        from tqdm import trange

        self.hyper_params.update(kwargs)

        logging.info("Starting Training Loop...")
        epochs_range = trange(num_epochs)
        for epoch in epochs_range:
            for i, data in enumerate(self.dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                ## Train with all-real batch
                self.D.net.zero_grad()
                real_cpu = data[0].to(self.hyper_params["device"])  # Format batch
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.hyper_params["device"])
                output = self.D.net(real_cpu).view(-1)  # Forward pass real batch through D
                err_d_real = self.criterion(output, label)  # Calculate loss on all-real batch
                err_d_real.backward()  # Calculate gradients for D in backward pass
                D_x = output.mean().item()

                ## Train with all-fake batch
                noise = torch.randn(b_size, self.hyper_params["nz"], 1, 1,
                                    device=self.hyper_params["device"])  # Generate batch of latent vectors
                fake = self.G.net(noise)  # Generate fake image batch with G
                label.fill_(self.fake_label)
                output = self.D.net(fake.detach()).view(-1)  # Classify all fake batch with D
                err_d_fake = self.criterion(output, label)  # Calculate D's loss on the all-fake batch
                err_d_fake.backward()  # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                d_g_z1 = output.mean().item()
                err_d = err_d_real + err_d_fake  # Compute error of D as sum over the fake and the real batches
                self.D.optimizer.step()  # Update D

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.G.net.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.D.net(fake).view(
                    -1)  # Since we just updated D, perform another forward pass of all-fake batch through D
                err_g = self.criterion(output, label)  # Calculate G's loss based on this output
                err_g.backward()  # Calculate gradients for G
                d_g_z2 = output.mean().item()
                self.G.optimizer.step()  # Update G

                # Update tqdm bar
                epochs_range.set_description(f"[({epoch + self.G.epoch_offset}&{epoch + self.D.epoch_offset}), {i}]\t"
                                             f"Loss_D: {err_d.item():.4f}\tLoss_G: {err_g.item():.4f}\tD(x): {D_x:.4f}\t"
                                             f"D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}")
                epochs_range.refresh()

                # Save Losses
                self.G.losses.append(err_g.item())
                self.D.losses.append(err_d.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % 10 == 0) or (epoch == num_epochs - 1):
                self.G.save(epoch, ckpt_dir)
                self.D.save(epoch, ckpt_dir)

    def printTraining(self):
        logging.info("Print training")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G.losses, label="G")
        plt.plot(self.D.losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def paint(self):
        logging.info("Paint random noise")

        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        noise = torch.randn(1, self.hyper_params["nz"], 1, 1, device=self.hyper_params["device"])
        image = self.G.net(noise).detach().numpy()[0]

        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(image))
        plt.show()
