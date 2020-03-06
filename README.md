# IBA: Informational Bottlenecks for Attribution

This repository contains an easy-to-use implementation for the IBA attribution method.
See our paper for a description: ["Restricting the Flow: Information Bottlenecks for Attribution"](https://openreview.net/forum?id=S1xWh1rYwB). We provide a TensorFlow and a PyTorch implementation.

In the notebook [example.ipynb](example.ipynb), the Per-Sample Bottleneck is
applied to pretrained ImageNet networks.

A short usage-description:

```python
# Create the Per-Sample Bottleneck:
btln = IBA()

# Add it to your model
btln.attach(model.conv4)

# Estimate the mean and variance.
btln.estimate(model, datagen)

# Closure that returns the loss for one batch
model_loss_closure = lambda x: -torch.log_softmax(model(x), 1)[:, target].mean()

# Create the heatmap
heatmap = btln.heatmap(img[None].to(dev), model_loss_closure)

# If you train your model, input distribution changes and you have to re-estimate the mean and std.
train(model)
btln.reset_estimate()
btln.estimate(model, datagen)
```

## Installation

You can either install it directly from git:
```bash
$ pip install git+https://github.com/berleon/IBA
```
To install the dependencies for `torch`, `tensorflow`, `tensorflow-gpu` or developement `dev`,
use:
```bash
$ pip install git+https://github.com/berleon/IBA[torch, dev]
```

You can also clone the repository locally and then install the development
version:
```bash
$ git clone https://github.com/attribution-bottleneck/per-sample-bottleneck
$ cd per-sample-bottlneck
$ pip install .
```

## Supported PyTorch and TensorFlow versions

We support tensorflow from `1.12.0` to `1.15.0`.
You might also be able to use our code from tensorflow 2 using the backward capatibility.
We currently not plan to support tensorflow 2.

## Reference

If you use this software for a scientific publication, please cite our paper:

```bibtex
@inproceedings{
Schulz2020Restricting,
title={Restricting the Flow: Information Bottlenecks for Attribution},
author={Karl Schulz and Leon Sixt and Federico Tombari and Tim Landgraf},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=S1xWh1rYwB}
}
```
