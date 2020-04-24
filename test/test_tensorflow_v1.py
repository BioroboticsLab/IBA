# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import numpy as np

try:
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    tf.disable_v2_behavior()
    tf.disable_v2_tensorshape()
except ModuleNotFoundError:
    import tensorflow as tf


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D

from IBA.tensorflow_v1 import model_wo_softmax, IBACopy, IBACopyInnvestigate
from IBA.tensorflow_v1 import IBALayer, to_saliency_map


INPUT_SHAPE = (32, 32, 3)


def simple_model(with_iba, with_softmax=False):

    model = Sequential()

    model.add(Conv2D(8, (3, 3), padding='same', name='conv1',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='relu1'))
    model.add(Conv2D(8, (3, 3), padding='same', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))    # 8

    model.add(Conv2D(8, (3, 3), padding='same', name='conv3'))
    if with_iba:
        # add iba to model definition
        model.add(IBALayer(name='iba'))
    model.add(Activation('relu', name='relu3'))
    model.add(Conv2D(8, (3, 3), padding='same', name='conv4'))
    model.add(Activation('relu', name='relu4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

    model.add(Flatten())
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(64, name='fc1'))
    model.add(Activation('relu', name='relu5'))
    model.add(Dropout(0.5, name='dropout2'))
    if with_softmax:
        softmax = 'softmax'
    else:
        softmax = None
    model.add(Dense(10, name='fc2',  activation=softmax))
    return model


def test_iba_layer(tmpdir):
    K.clear_session()
    model = simple_model(with_iba=True, with_softmax=True)

    # check if softmax is removed
    model = model_wo_softmax(model)

    x = np.random.uniform(size=(10, 32, 32, 3))

    model.predict(x)

    iba = model.get_layer(name='iba')  # type: IBALayer

    # fit a single sample
    iba.fit({model.input: x})

    # fit a generator
    def generator():
        for i in range(10):
            x = np.random.uniform(size=(2, 32, 32, 3))
            yield {model.input: x}

    hyperparams = {'batch_size': 10, 'steps': 10, 'beta': 1, 'learning_rate': 1,
                   'min_std': 0.01, 'smooth_std': 1, 'normalize_beta': True}
    iba.set_default(**hyperparams)
    assert iba.get_default() == hyperparams

    iba.fit_generator(generator(), n_samples=15)

    # setup model loss
    iba.set_classification_loss(model.output)

    # analyze
    x = np.random.uniform(size=(1, 32, 32, 3))
    iba.analyze({model.input: x, iba.target: np.array([4])})

    iba.collect_all()
    iba.analyze({model.input: x, iba.target: np.array([4])})
    report = iba.get_report()
    assert 'alpha' in report['init']
    assert 'alpha' in report['final']
    assert 'loss' in report[0]

    iba.collect('loss')
    capacity = iba.analyze({model.input: x, iba.target: np.array([4])})
    saliency_map = to_saliency_map(capacity, shape=(32, 32))
    assert saliency_map.shape == (32, 32)
    report = iba.get_report()
    assert 'alpha' not in report['init']

    # for the test, remove all information
    capacity = iba.analyze({model.input: x, iba.target: np.array([4])}, beta=100000)

    print("STATE", iba._get_session().run([iba._restrict_flow, iba._use_layer_input]))
    logits = model.predict(x)
    with iba.restrict_flow():
        print("STATE 2", iba._get_session().run([iba._restrict_flow, iba._use_layer_input]))
        # computes logits using only a subset of all information
        logits_restricted = model.predict(x)

    # logits should be considerable different
    assert np.abs(logits - logits_restricted).mean() > 1e-3


def test_iba_layer_1d(tmpdir):
    K.clear_session()
    model = Sequential()

    model.add(Conv1D(32, 3, padding='same', name='conv1',
                     input_shape=(32, 3)))
    model.add(Activation('relu', name='relu1'))
    model.add(Conv1D(32, 3, padding='same', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling1D(pool_size=2, name='pool2'))    # 8

    model.add(Conv1D(32, 3, padding='same', name='conv3'))
    # add iba to model definition
    model.add(IBALayer(name='iba'))
    model.add(Activation('relu', name='relu3'))
    model.add(Conv1D(32, 3, padding='same', name='conv4'))
    model.add(Activation('relu', name='relu4'))
    model.add(MaxPooling1D(pool_size=2, name='pool4'))

    model.add(Flatten())
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(256, name='fc1'))
    model.add(Activation('relu', name='relu5'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(10, name='fc2'))

    x = np.random.uniform(size=(10, 32, 3))

    model.predict(x)

    iba = model.get_layer(name='iba')
    iba.fit({model.input: x})
    iba.set_classification_loss(model.output)

    x = np.random.uniform(size=(1, 32, 3))
    iba.analyze({model.input: x, iba.target: np.array([4])})


def test_copy_graph_innvestigate(tmpdir):
    INPUT_SHAPE = (32, 32, 3)

    def random_input_generator():
        while True:
            yield np.random.uniform(size=(5,) + INPUT_SHAPE), np.random.normal(size=5) > 0

    K.clear_session()
    model = simple_model(with_iba=False)
    feat_layer = model.get_layer(name='conv3')

    x = np.random.normal(size=(1,) + INPUT_SHAPE)
    logits_before = model.predict(x)

    analyzer = IBACopyInnvestigate(model, feature_name=feat_layer.output.name)

    logits_after_copy = model.predict(x)
    assert (logits_before == logits_after_copy).all()

    analyzer.fit_generator(random_input_generator(), steps_per_epoch=2)
    analyzer.analyze(np.random.normal(size=(1, ) + INPUT_SHAPE))

    fname = str(tmpdir.join('innvestigate.npz'))
    analyzer.save_npz(fname)

    load_graph = tf.Graph()
    sess = tf.Session(graph=load_graph)
    with sess.as_default(), load_graph.as_default():
        analyzer_loaded = IBACopyInnvestigate.load_npz(fname)

    x = np.random.normal(size=(1,) + INPUT_SHAPE)
    logit_copied = analyzer.predict(x)
    with sess.as_default(), load_graph.as_default():
        logit_loaded = analyzer_loaded.predict(x)
    logit_model = model.predict(x)

    assert np.abs(logit_model - logit_copied).mean() < 1e-5
    assert np.abs(logit_model - logit_loaded).mean() < 1e-5


def test_copy_graph_raw():
    K.clear_session()
    model = simple_model(with_iba=False)
    feat_layer = model.get_layer(name='conv2')

    shape = [int(d) for d in feat_layer.output.shape[1:]]
    mean = np.random.uniform(size=shape)
    std = np.random.uniform(size=shape)

    analyzer = IBACopy(feat_layer.output, [model.output.name],
                       feature_mean=mean, feature_std=std,
                       feature_active=np.ones_like(std))
    analyzer.assert_variables_equal()

    x = np.random.normal(size=(1, ) + INPUT_SHAPE)

    # check weights are copied, i.e. same prediction
    logit_model = model.predict(x)
    logit_copied = analyzer.predict({model.input: x})
    assert np.abs(logit_copied - logit_model).mean() < 1e-5

    analyzer._outputs[0]
    analyzer.set_classification_loss()
    analyzer.analyze({model.input: x}, {analyzer.target: np.array([30])})

    kernel = K.get_session().run(feat_layer.kernel)
    new_kernel = kernel + np.ones_like(kernel)

    K.get_session().run(tf.assign(feat_layer.kernel, new_kernel))

    with pytest.raises(AssertionError):
        analyzer.assert_variables_equal()

    analyzer.update_variables()
    analyzer.assert_variables_equal()
