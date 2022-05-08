from monet import Monet


monetPainter = Monet(
    pretrained_generator=None,
    pretrained_discriminator=None,
    training_dataset="../../dataset",
)
monetPainter.train(100, "/User/Documents/HomeWork/MonetGAN")
monetPainter.paint()
