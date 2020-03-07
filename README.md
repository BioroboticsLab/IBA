# IBA: Informational Bottlenecks for Attribution

This repository contains an easy-to-use implementation for the IBA attribution method.
See our paper for a description: ["Restricting the Flow: Information Bottlenecks for Attribution"](https://openreview.net/forum?id=S1xWh1rYwB). We provide a [TensorFlow v1](https://www.tensorflow.org/) and a [PyTorch](https://pytorch.org/) implementation.


Examplary usage (PyTorch):

```python
from IBA.pytorch import IBA
from IBA.utils import plot_saliency_map

model = Net()
# Create the Per-Sample Bottleneck:
iba = IBA(model.conv4)

# Estimate the mean and variance.
iba.estimate(model, datagen)

img, target = next(iter(datagen(batch_size=1)))

# Closure that returns the loss for one batch
model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

# Explain class target for the given image
salienct_map = iba.analyze(img.to(dev), model_loss_closure)
plot_saliency_map(img.to(dev))
```


## Documentation

The API documentation is hosted here.

TODO: mention the different notebooks

## Installation

You can either install it directly from git:
```bash
$ pip install git+https://github.com/berleon/IBA
```

To install the dependencies for `torch`, `tensorflow`, `tensorflow-gpu` or developement `dev`,
use the following syntax:
```bash
$ pip install git+https://github.com/berleon/IBA[torch, dev]
```

For development,yYou can also clone the repository locally and then install in development
mode:
```bash
$ git clone https://github.com/attribution-bottleneck/per-sample-bottleneck
$ cd per-sample-bottlneck
$ pip install -e .
```

## Supported PyTorch and TensorFlow versions

We support tensorflow from `1.12.0` to `1.15.0`.
Although we currently not plan to support tensorflow 2,
it might be possible to use our code from tensorflow 2 using the backward capatibility wrapper.

For PyTorch, we support version `1.1.0` to `1.4.0`.

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
