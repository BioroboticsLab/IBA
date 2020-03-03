# Per-Sample Bottleneck

This repository contains an easy-to-use pytorch implementation for the Per-Sample Bottleneck for
attribution. See our paper for more detail: ["Restricting the Flow: Information Bottlenecks for Attribution"](https://openreview.net/forum?id=S1xWh1rYwB). In the notebook [example.ipynb](example.ipynb), the Per-Sample Bottleneck is
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
$ pip install git+https://github.com/attribution-bottleneck/per-sample-bottleneck
```

Or clone this repository locally and then install it:
```bash
$ git clone https://github.com/attribution-bottleneck/per-sample-bottleneck
$ cd per-sample-bottlneck
$ pip install .
```

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
