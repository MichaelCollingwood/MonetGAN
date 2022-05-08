from monet import Monet


print("start")
monetPainter = Monet(
    pretrained_generator=None,
    pretrained_discriminator=None,
    training_dataset="../../dataset",
)
print("milestone A")
monetPainter.train(3, "")
print("milestone B")
print(monetPainter.paint())

