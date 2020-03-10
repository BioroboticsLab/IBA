# IBA: Informational Bottlenecks for Attribution


[[Paper Arxiv]](https://arxiv.org/abs/2001.00396)
| [[Paper Code]](https://github.com/BioroboticsLab/IBA-paper-code)
| [[Reviews]](https://openreview.net/forum?id=S1xWh1rYwB)
| [[API Documentation]](https://iba.readthedocs.io/en/latest/)
| [[Examples]](notebooks)
| [[Installation]](#installation)

[![Build Status](https://travis-ci.com/BioroboticsLab/IBA.svg?branch=master)](https://travis-ci.com/BioroboticsLab/IBA)
[![Documentation Status](https://readthedocs.org/projects/iba/badge/?version=latest)](https://iba.readthedocs.io/en/latest/?badge=latest)


<p align="center">
    <img alt="Example GIF" width="300" src="https://github.com/BioroboticsLab/IBA-paper-code/raw/master/monkeys.gif"><br>
    Iterations of the Per-Sample Bottleneck
</p>


This repository contains an easy-to-use implementation for the IBA attribution method.
See our paper for a description: ["Restricting the Flow: Information
Bottlenecks for Attribution"](https://openreview.net/forum?id=S1xWh1rYwB). We
provide a [TensorFlow v1](https://www.tensorflow.org/) and
a [PyTorch](https://pytorch.org/) implementation.

For the code to reproduce our paper, see [IBA-paper-code](https://github.com/BioroboticsLab/IBA-paper-code).


## PyTorch

Examplary usage:
```python
from IBA.pytorch import IBA, plot_saliency_map

model = Net()
# Create the Per-Sample Bottleneck:
iba = IBA(model.conv4)

# Estimate the mean and variance.
iba.estimate(model, datagen)

img, target = next(iter(datagen(batch_size=1)))

# Closure that returns the loss for one batch
model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

# Explain class target for the given image
saliency_map = iba.analyze(img.to(dev), model_loss_closure)
plot_saliency_map(img.to(dev))
```


## Tensorflow

```python
from IBA.tensorflow_v1 import IBACopyInnvestigate, plot_saliency_map

# load model
model_softmax = VGG16(weights='imagenet')

# remove the final softmax layer
model = model_wo_softmax(model_softmax)

# select layer after which the bottleneck will be inserted
feat_layer = model.get_layer(name='block3_conv2')

# copies the model
iba = IBACopyInnvestigate(
    model,
    neuron_selection_mode='index',
    feature_name=feat_layer.output.name,
)

# estimate feature mean and std
iba.fit_generator(image_generator(), steps_per_epoch=50)

# get the saliency map and plot
saliency_map = iba.analyze(monkey, neuron_selection=monkey_target)
plot_saliency_map(saliency_map, img=norm_image(monkey[0]))
```

**Table:** Overview over the different tensorflow classes.
**(Task)** type of task (i.e. regression, classification, unsupervised).
**(Layer)** requires you to add a layer to the explained model.
**(Copy)** copies the tensorflow graph.

| Class | Task | Layer | Copy | Remarks
|-------|------|-------|------|--------
| [`IBALayer`](https://iba.readthedocs.io/en/latest/api/iba_tensorflow_v1.html#IBA.tensorflow_v1.IBALayer) | Any | ✅  | ❌ | Recommended              |
| [`IBACopy`](https://iba.readthedocs.io/en/latest/api/iba_tensorflow_v1.html#IBA.tensorflow_v1.IBACopy)| Any | ❌ | ✅ | Very flexible
| [`IBACopy`](https://iba.readthedocs.io/en/latest/api/iba_tensorflow_v1.html#IBA.tensorflow_v1.IBACopyInnvestigate)| Classification | ❌ | ✅ |  Nice API for classification


## Documentation

[[PyTorch API]](https://iba.readthedocs.io/en/latest/api/iba_pytorch.html)
| [[TensorFlow API]](https://iba.readthedocs.io/en/latest/api/iba_tensorflow_v1.html)

The API documentation is hosted [here](https://iba.readthedocs.io/en/latest).

**Table:** Examplary jupyter notebooks


| Notebook | Description |
|----------|-------------|
| [pytorch_IBA.ipynb](notebooks/pytorch_IBA.ipynb) | Per-Sample Bottleneck |
| [pytorch_IBA_train_readout.ipynb](notebooks/pytorch_IBA_train_readout.ipynb) | Train a Readout Bottleneck |
| [tensorflow_IBALayer_cifar.ipynb](notebooks/tensorflow_IBALayer_cifar.ipynb) | Train a CIFAR model containing an IBALayer |
| [tensorflow_IBACopy_imagenet.ipynb](notebooks/tensorflow_IBACopy_imagenet.ipynb) | Explains a ImageNet model |
| [tensorflow_IBACopyInnvestigate_imagenet.ipynb](notebooks/tensorflow_IBACopyInnvestigate_imagenet.ipynb)| [innvestigate](https://github.com/albermax/innvestigate) api wrapper |


## Installation

You can install it directly from git:

```bash
$ pip install git+https://github.com/BioroboticsLab/IBA
```

To install the dependencies for `torch`, `tensorflow`, `tensorflow-gpu` or developement `dev`,
use the following syntax:
```bash
$ pip install git+https://github.com/BioroboticsLab/IBA[torch, dev]
```

For development, you can also clone the repository locally and then install in development
mode:
```bash
$ git clone https://github.com/BioroboticsLab/IBA
$ cd per-sample-bottlneck
$ pip install -e .
```

**Table:** Supported versions

|Package| From | To |
|-------|------|----|
| TensorFlow | `1.12.0` | `1.15.0` |
| PyTorch | `1.1.0` | `1.4.0` |


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
