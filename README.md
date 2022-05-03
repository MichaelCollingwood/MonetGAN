# **Monet Painter Package**

This package provides a wrapper class for a machine learning model. It exposes training and inference methods.

## **Model Architecture**

Based on a DCGAN neural network (Ian Goodfellow, 2014)

_TODO_: 
- Expand to a stylised GAN ([BlendGAN](https://arxiv.org/pdf/2110.11728.pdf))
- _TODO_: Investigate Growing Neural Network (GradMax, NeST)

## **Training**


```
monetPainter.train(
    training_config,
    save_checkpoints="ckpts/",
)
```

## **Inference**

```
monetPainter.paint(
    random_seed,
)
```

_TODO_:
- Paint by mood using stylised GAN & averaged $z_s$ latent vector of clustered moody style images.
- Broaden MonetPainter $\rightarrow$ Painter. Specify styles of painters to use. Also predict painters using transfer-learnt discriminator.
