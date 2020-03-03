import pytest
import numpy as np

import tensorflow.compat.v1 as tf

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D


from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16

from IBA.tensorflow_v1 import model_wo_softmax, IBACopyGraph, IBACopyGraphInnvestigate, IBALayer


def simply_model():
    pass


@pytest.fixture
def vgg16():
    K.clear_session()
    model_softmax = VGG16(weights='imagenet')
    model = model_wo_softmax(model_softmax)
    return model


@pytest.fixture
def resnet50():
    K.clear_session()
    model_softmax = ResNet50(weights='imagenet')
    model = model_wo_softmax(model_softmax)
    return model


def test_iba_layer(tmpdir):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='conv1',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='relu1'))
    model.add(Conv2D(32, (3, 3), padding='same', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))    # 8

    model.add(Conv2D(32, (3, 3), padding='same', name='conv3'))
    # add iba to model definition
    model.add(IBALayer(name='iba'))
    model.add(Activation('relu', name='relu3'))
    model.add(Conv2D(32, (3, 3), padding='same', name='conv4'))
    model.add(Activation('relu', name='relu4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

    model.add(Flatten())
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(256, name='fc1'))
    model.add(Activation('relu', name='relu5'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(10, name='fc2'))

    x = np.random.uniform(size=(10, 32, 32, 3))

    model.predict(x)

    iba = model.get_layer(name='iba')
    iba.fit({model.input: x})
    iba.set_classification_loss(model.output)


    x = np.random.uniform(size=(1, 32, 32, 3))
    iba.analyze({model.input: x, iba.target: np.array([4])})

    iba.collect_all()
    iba.analyze({model.input: x, iba.target: np.array([4])})
    report = iba.get_report()
    assert 'alpha' in report['init']
    assert 'alpha' in report['final']
    assert 'loss' in report[0]

    iba.collect('loss')
    iba.analyze({model.input: x, iba.target: np.array([4])})
    report = iba.get_report()
    assert 'alpha' not in report['init']


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


def test_copy_graph_innvestigate(vgg16, tmpdir):
    def generator():
        while True:
            yield np.random.uniform(size=(5, 224, 224, 3)), np.random.normal(size=5) > 0

    feat_layer = vgg16.get_layer(name='block4_conv1')
    analyzer = IBACopyGraphInnvestigate(vgg16, feature_name=feat_layer.output.name)
    analyzer.fit_generator(generator(), steps_per_epoch=2)
    analyzer.analyze(np.random.normal(size=(1, 224, 224, 3)))

    fname = str(tmpdir.join('innvestigate.npz'))
    analyzer.save_npz(fname)

    load_graph = tf.Graph()
    sess = tf.Session(graph=load_graph)
    with sess.as_default(), load_graph.as_default():
        analyzer_loaded = IBACopyGraphInnvestigate.load_npz(fname)

    x = np.random.normal(size=(1, 224, 224, 3))
    logit_copied = analyzer.predict({vgg16.input: x})
    with sess.as_default(), load_graph.as_default():
        logit_loaded = analyzer_loaded.predict({analyzer_loaded._model.input: x})
    logit_model = vgg16.predict(x)

    assert np.abs(logit_model - logit_copied).mean() < 1e-6
    assert np.abs(logit_model - logit_loaded).mean() < 1e-6


def test_copy_graph_raw(resnet50):
    def generator():
        while True:
            yield np.random.uniform(size=(5, 224, 224, 3)), np.random.normal(size=5) > 0

    for layer in resnet50.layers:
        #print(layer.name)
        pass

    feat_layer = resnet50.get_layer(name='conv3_block4_out')


    shape = [int(d) for d in feat_layer.output.shape[1:]]
    mean = np.random.uniform(size=shape)
    std = np.random.uniform(size=shape)

    analyzer = IBACopyGraph(feat_layer.output.name, [resnet50.output.name],
                            feature_mean_std=[mean, std])
    x = np.random.normal(size=(1, 224, 224, 3))

    # check weights are copied, i.e. same prediction
    logit_model = resnet50.predict(x)
    logit_copied = analyzer.predict({resnet50.input: x})
    assert np.abs(logit_copied - logit_model).mean() < 1e-6

    analyzer._outputs[0]
    analyzer.set_classification_loss()
    analyzer.analyze({resnet50.input: x}, {analyzer.target: np.array([30])})
