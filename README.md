# Per-Sample Bottleneck

This repository contains an easy-to-use pytorch implementation for the Per-Sample Bottleneck for
attribution.  In the notebook [example.ipynb](example.ipynb), the Per-Sample Bottleneck is
applied to pretrained ImageNet networks.

A short usage-description:

```python
# Create the Per-Sample Bottleneck:
btln = PerSampleBottleneck(channels, height, width)

# Add it to your model
model.conv4 = nn.Sequential(model.conv4, btln)

# Estimate the mean and variance
btln.estimate(model, datagen)

# Closure that returns the loss for one batch
model_loss_closure = lambda x: -torch.log_softmax(model(x), 1)[:, target].mean()

# Create the heatmap
heatmap = btln.heatmap(img[None].to(dev), model_loss_closure)
```

## Installation

You can either install it directly from git:
```bash
$ pip install git+https://github.com/attribution-bottleneck/per-sample-bottleneck
```

Or clone this repository locally and then install it:
```bash
$ git clone https://github.com/attribution-bottleneck/per-sample-bottleneck
$ cd per-sample-bottlneck
$ pip install .
```
