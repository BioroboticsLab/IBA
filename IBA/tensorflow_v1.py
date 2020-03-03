from collections import OrderedDict
from contextlib import contextmanager

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import numpy as np
import keras
from IBA.utils import WelfordEstimator
import keras.backend as K
from tqdm import trange
from tqdm.auto import tqdm
from IBA._keras_graph import _BatchSequence, contains_activation

# expose for importing
from IBA._keras_graph import model_wo_softmax   # noqa


tf.disable_v2_behavior()


class TFWelfordEstimator(WelfordEstimator):
    def __init__(self, feature_name, graph=None):
        self._feature_name = feature_name
        self._graph = graph or tf.get_default_graph()
        super().__init__()

    def fit(self, feed_dict, session=None, run_kwargs={}):
        session = session or tf.get_default_session() or K.get_session()
        feature = self._graph.get_tensor_by_name(self._feature_name)
        feat_values = session.run(feature, feed_dict=feed_dict, **run_kwargs)
        super().fit(feat_values)

    def fit_generator(self, generator, session=None, progbar=True, run_kwargs={}):
        for feed_dict in tqdm(generator, progbar=progbar):
            self.fit(feed_dict, session, run_kwargs)

    def state_dict(self):
        state = super().state_dict()
        state['feature_name'] = self._feature_name

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._feature_mean = state['feature_mean']


def kl_div(r, lambda_, mean_r, std_r):
    r_norm = (r - mean_r) / std_r
    var_z = (1 - lambda_) ** 2

    log_var_z = tf.log(var_z)

    mu_z = r_norm * lambda_

    capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
    return capacity


def gaussian_kernel(size, std):
    """Makes 2D gaussian Kernel for convolution."""
    d = tfp.distributions.Normal(0., std)
    vals = d.prob(tf.cast(tf.range(start=-size, limit=size + 1), tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def gaussian_blur(x, std=1.):
    # Cover 2.5 stds in both directions
    kernel_size = tf.cast((tf.round(2 * std)) * 2 + 1, tf.int32)

    kernel = gaussian_kernel(kernel_size // 2, std)
    kernel = kernel[:, :, None, None]
    kernel = tf.tile(kernel, (1, 1, x.shape[-1], 1))

    kh = kernel_size//2

    if len(x.shape) == 4:
        x = tf.pad(x, [[0, 0], [kh, kh], [kh, kh], [0, 0]], "REFLECT")
        x_blur = tf.nn.depthwise_conv2d(
            x,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='blurring',
        )
    elif len(x.shape) == 3:
        x = tf.pad(x, [[0, 0], [kh, kh], [0, 0]], "REFLECT")
        kernel = kernel[:, kh+1:kh+2]
        x_extra_dim = x[:, :, None]
        print('x_extra_dim', x_extra_dim.shape, kernel.shape)
        x_blur = tf.nn.depthwise_conv2d(
            x_extra_dim,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='blurring',
        )
        x_blur = x_blur[:, :, 0]
    else:
        raise ValueError("shape not supported! got: {}".format(x.shape))
    return tf.cond(tf.math.equal(std, 0.), lambda: x, lambda: x_blur)


class IBALayer(keras.layers.Layer):
    def __init__(self, estimator=None, batch_size=10,
                 feature_mean_std=None,
                 **kwargs):
        self._estimator = estimator
        self._model_loss_set = False

        if feature_mean_std is not None:
            self._feature_mean_std_given = True
            self._feature_mean = feature_mean_std[0]
            self._feature_std = feature_mean_std[1]
        else:
            self._feature_mean_std_given = False
            self._feature_mean = None
            self._feature_std = None

        self._collect_names = []

        self._report_tensors = OrderedDict()
        self._report_tensors_first = OrderedDict()

        super().__init__(**kwargs)

    # Reporting

    def _report(self, name, tensor):
        assert name not in self._report_tensors
        self._report_tensors[name] = tensor

    def _report_first(self, name, tensor):
        assert name not in self._report_tensors_first
        self._report_tensors_first[name] = tensor

    def _get_report_tensors(self):
        ret = OrderedDict()
        for name in self._collect_names:
            if name in self._report_tensors:
                ret[name] = self._report_tensors[name]
        return ret

    def _get_report_tensors_first(self):
        ret = self._get_report_tensors()
        for name in self._collect_names:
            if name in self._report_tensors_first:
                ret[name] = self._report_tensors_first[name]
        return ret

    def collect(self, *names):
        for name in names:
            assert name in self._report_tensors or name in self._report_tensors_first, \
                "not tensor found with name {}! Try one of these: {}".format(
                    name, self.available_report_variables())
        self._collect_names = names

    def collect_all(self):
        self.collect(*self.available_report_variables())

    def available_report_variables(self):
        return sorted(list(self._report_tensors.keys()) + list(self._report_tensors_first.keys()))

    def get_report(self):
        return self._log

    # Build of layer
    def build(self, input_shape):
        self._featuremap_shape = [1, ] + [int(d) for d in input_shape[1:]]
        shape = self._featuremap_shape
        print('build shape', shape)

        k = np.prod([int(s) for s in shape]).astype(np.float32)
        # optimization placeholders
        self._learning_rate = tf.get_variable('learning_rate', dtype=tf.float32, initializer=1.)
        self._beta = tf.get_variable('beta', dtype=tf.float32,
                                     initializer=np.array(10./k).astype(np.float32))

        self._batch_size = tf.get_variable('batch_size', dtype=tf.int32, initializer=10)

        # trained parameters
        alpha_init = 5
        self._alpha = tf.get_variable(name='alpha', initializer=alpha_init*tf.ones(shape))

        # feature map
        self._featuremap = tf.get_variable('featuremap', shape, trainable=False)

        # std normal noise
        self._std_normal = tf.random.normal([self._batch_size, ] + shape[1:])

        # mean of feature map r
        self._mean_r = tf.get_variable(
            name='mean_r', trainable=False,  dtype=tf.float32, initializer=tf.zeros(shape))
        # std of feature map r
        self._std_r = tf.get_variable(
            name='std_r', trainable=False,  dtype=tf.float32, initializer=tf.zeros(shape))

        # mask that indicate that no noise should be applied to a specific neuron
        self._pass_mask = tf.get_variable(name='pass_mask', trainable=False,
                                          dtype=tf.float32, initializer=tf.zeros(shape))

        # flag to restrict the flow
        self._restrict_flow = tf.get_variable(name='restrict_flow', trainable=False,
                                              dtype=tf.bool, initializer=False)

        self._use_layer_input = tf.get_variable(name='use_layer_input', trainable=False,
                                                dtype=tf.bool, initializer=False)

        # min standard derivation per neuron
        self._min_std_r = tf.get_variable('min_std_r', dtype=tf.float32, initializer=0.1)
        # kernel size for gaussian blur
        self._smooth_std = tf.get_variable('smooth_std', dtype=tf.float32, initializer=1.)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if self._estimator is None and not self._feature_mean_std_given:
            self._estimator = TFWelfordEstimator(inputs.name)

        # shape = [1,] + inputs.shape[1:].as_list()
        # self.active_neurons = tf.get_variable('active_neurons', shape, trainable=False)

        featuremap = tf.cond(self._use_layer_input,
                             lambda: inputs,
                             lambda: self._featuremap)

        tile_shape = [self._batch_size, ] + [1] * (len(self._featuremap_shape) - 1)
        R = tf.tile(featuremap, tile_shape)
        pass_mask = tf.tile(self._pass_mask, tile_shape)
        restrict_mask = 1 - pass_mask

        std_r_min = tf.maximum(self._std_r, self._min_std_r)
        std_r_min = tf.clip_by_value(std_r_min, np.exp(-10), np.exp(10))

        # std_too_low = tf.cast(self.std_x < self.min_std_r, tf.float32)
        # std_large_enough = tf.cast(self.std_x >= self.min_std_r, tf.float32)

        lambda_pre_blur = tf.sigmoid(self._alpha)
        λ = gaussian_blur(lambda_pre_blur, std=self._smooth_std)

        # ε ~ N(μ_r, σ_r)
        ε = std_r_min * self._std_normal + self._mean_r

        Z = λ * R + (1 - λ) * ε

        # let all information through for neurons in pass_mask
        Z_with_passing = restrict_mask * Z + pass_mask * R

        output = tf.cond(self._restrict_flow, lambda: Z_with_passing, lambda: inputs)
        # save capacityies
        self._capacity = kl_div(R, λ, self._mean_r, std_r_min) * restrict_mask
        self._capacity_in_bits = self._capacity / np.log(2)
        self._capacity_mean = tf.reduce_sum(self._capacity) / tf.reduce_sum(restrict_mask)

        # save tensors for report
        self._report('lambda_pre_blur', lambda_pre_blur)
        self._report('lambda',  λ)
        self._report('eps', ε)
        self._report('alpha', self._alpha)
        self._report('capacity', self._capacity)
        self._report('capacity_in_bits', self._capacity_in_bits)
        self._report('capacity_mean', self._capacity_mean)

        self._report('perturbed_featuremap', Z)
        self._report('perturbed_featuremap_with_passing', Z_with_passing)
        self._report_first('featuremap', featuremap)
        self._report_first('pass_mask', pass_mask)

        self._report_first('featuremap_mean', self._mean_r)
        self._report_first('featuremap_std', self._std_r)

        # restrict_flow = tf.cast(self._restrict_flow, tf.float32)
        # return restrict_flow * Z_with_passing + (1 - restrict_flow) * inputs
        return output

    def _get_session(self, session=None):
        return session or keras.backend.get_session()

    # Set model loss

    def set_classification_loss(self, logits, optimizer_cls=tf.train.AdamOptimizer):
        """
        Creates a cross-entropy loss from logits.
        """
        self.target = tf.get_variable('iba_target', dtype=tf.int32, initializer=[1])

        target_one_hot = tf.one_hot(self.target, depth=logits.shape[-1])
        loss_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_one_hot,
            logits=logits,
            name='cross_entropy'
        )
        loss_ce_mean = tf.reduce_mean(loss_ce)
        self._report('logits', logits)
        self._report('cross_entropy', loss_ce)
        self.set_model_loss(loss_ce_mean, optimizer_cls)
        return self.target

    def set_model_loss(self, model_loss, optimizer_cls=tf.train.AdamOptimizer):
        self._optimizer = optimizer_cls(learning_rate=self._learning_rate)
        information_loss = self._beta * self._capacity_mean
        loss = model_loss + information_loss
        self._optimizer_step = self._optimizer.minimize(loss, var_list=[self._alpha])

        self._report('loss', loss)
        self._report('model_loss', model_loss)
        self._report('information_loss', information_loss)
        self._report('grad_loss_wrt_alpha', tf.gradients(loss, self._alpha)[0])
        self._model_loss_set = True

    # Fit std and mean estimator
    def fit(self, feed_dict, session=None, run_kwargs={}):
        self._estimator.fit(feed_dict, session, run_kwargs)

    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, session=None):
        """

        """
        session = self._get_session(session)
        for epoch in range(epochs):
            for step, (imgs, targets) in enumerate(tqdm(
                generator, disable=verbose, desc="[Fit Estimator #{}]".format(epoch))):
                feed_dict = {self._model.input: imgs}
                self._estimator.fit(feed_dict, session=session)
                if step + 1 >= steps_per_epoch:
                    break

    def analyze(self, feed_dict,
                pass_mask=None,
                batch_size=10,
                steps=10,
                beta=10,
                learning_rate=100,
                min_std=0.01,
                smooth_std=1,
                normalize_beta=True,
                session=None,
                progbar=False):
        session = self._get_session(session)
        featuremap = session.run(self.input, feed_dict=feed_dict)

        return self._analyze_featuremap(
            featuremap,
            feed_dict,
            pass_mask=pass_mask,
            batch_size=batch_size,
            steps=steps,
            beta=beta,
            learning_rate=learning_rate,
            min_std=min_std,
            smooth_std=smooth_std,
            normalize_beta=normalize_beta,
            session=session,
            progbar=False)

    def _analyze_featuremap(self, featuremap,
                            feed_dict,
                            pass_mask=None,
                            batch_size=10,
                            steps=10,
                            beta=10,
                            learning_rate=100,
                            min_std=0.01,
                            smooth_std=1,
                            normalize_beta=True,
                            session=None,
                            progbar=False):
        if session is None:
            session = keras.backend.get_session()

        if not hasattr(self, '_optimizer'):
            raise ValueError("Optimizer not build yet! You have to specify your model loss "
                             "by calling the set_model_loss method.")
        self._log = OrderedDict()

        if not normalize_beta:
            # we use the mean of the capacity, which is equivalent to dividing by k=h*w*c.
            # therefore, we have to denormalize beta:
            beta = beta * np.prod(featuremap.shape)

        if not self._feature_mean_std_given:
            assert self._estimator.n_samples() > 0

            self._feature_mean = self._estimator.mean()
            self._feature_std = self._estimator.std()
        else:
            assert self._estimator is None or self._estimator.n_samples() == 0, \
                "Estimator was fitted but you also provided feature_mean_std!"

        def maybe_unsqueeze(x):
            if len(self._mean_r.shape) == len(x.shape) + 1:
                return x[None]
            else:
                return x
        self._feature_mean = maybe_unsqueeze(self._feature_mean)
        self._feature_std = maybe_unsqueeze(self._feature_std)

        # set hyperparameters
        assigns = [
            self._alpha.initializer,
            tf.variables_initializer(self._optimizer.variables()),
            tf.assign(self._mean_r, self._feature_mean, name='assign_feature_mean'),
            tf.assign(self._std_r, self._feature_std, name='assign_feature_std'),
            tf.assign(self._featuremap, featuremap, name='assign_feature'),
            tf.assign(self._beta, beta, name='assign_beta'),
            tf.assign(self._smooth_std, smooth_std, name='assign_smooth_std'),
            tf.assign(self._min_std_r, min_std, name='assign_min_std'),
            tf.assign(self._learning_rate, learning_rate, name='assign_lr'),
            tf.assign(self._restrict_flow, True, name='assign_restrict_flow'),
            tf.assign(self._use_layer_input, False, name='assign_use_layer_input'),
        ]
        if pass_mask is None:
            pass_mask = tf.zeros_like(self._pass_mask)
        assigns.append(tf.assign(self._pass_mask, pass_mask))
        session.run(assigns)

        report_tensors = self._get_report_tensors()
        report_tensors_first = self._get_report_tensors_first()

        if len(report_tensors_first) > 0:
            outs = session.run(list(report_tensors_first.values()),
                               feed_dict=feed_dict)
            self._log['init'] = OrderedDict(zip(report_tensors_first.keys(), outs))
        else:
            self._log['init'] = OrderedDict()

        for step in trange(steps, disable=not progbar):
            outs = session.run(
                [self._optimizer_step] + list(report_tensors.values()),
                feed_dict=feed_dict)
            self._log[step] = OrderedDict(zip(report_tensors_first.keys(), outs[1:]))

        final_report_tensors = list(report_tensors.values())
        final_report_tensor_names = list(report_tensors.keys())
        if 'capacity' not in report_tensors:
            final_report_tensors.append(self._capacity)
            final_report_tensor_names.append('capacity')

        vals = session.run(final_report_tensors, feed_dict=feed_dict)
        self._log['final'] = OrderedDict(zip(final_report_tensor_names, vals))

        return self._log['final']['capacity']

    def state_dict(self):
        return {
            'estimator': self._estimator.state_dict(),
            'feature_mean': self._feature_mean,
            'feature_std': self._feature_std,
        }

    def load_state_dict(self, state):
        self._estimator.load_state_dict(state['estimator'])
        self._feature_mean = state['feature_mean']
        self._feature_std = state['feature_std']


class IBACopyGraph(IBALayer):
    """
    Inject an IBALayer into an existing model by partially copying the model.
    IBACopyGraph is very useful for pretrained models which you cannot alter
    otherwise.  The main drawback is that changes to the original graph will
    have no effect on the explanations.  Adding IBALayer directly to our model
    might also be more memory efficient.
    """

    def __init__(self, feature_name, output_names,
                 estimator=None,
                 feature_mean_std=None,
                 graph=None,
                 session=None,
                 copy_session_config=None,
                 **keras_kwargs,
                 ):
        # The tensorflow graph is immutable. However, we have to add noise to get our
        # post-hoc explanation. If only the graph of the model is available, we copy the graph
        # and alter it using `import_graph_def`. This is not nice. I would love to use hooks.
        #
        # _original_*   refer to objects from the original session
        # all other variables refer to objects in the copied graph
        #
        if type(output_names) == str:
            output_names = [output_names]
        self._output_names = output_names
        self._feature_name = feature_name
        self._original_output_names = self._output_names
        self._original_graph = graph or tf.get_default_graph()
        self._original_graph_def = self._original_graph.as_graph_def()
        self._original_session = session or K.get_session()

        self._original_R = self._original_graph.get_tensor_by_name(feature_name)

        if copy_session_config is None:
            copy_session_config = tf.ConfigProto(allow_soft_placement=True)

        super().__init__(estimator=estimator,
                         feature_mean_std=feature_mean_std,
                         **keras_kwargs)
        if self._estimator is None and not self._feature_mean_std_given:
            self._estimator = TFWelfordEstimator(feature_name, graph=self._original_graph)
        # the new graph and session
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph, config=copy_session_config)

        with self.copied_session_and_graph_as_default():

            R = tf.get_variable('replaced_feature_map', dtype=tf.float32,
                                initializer=tf.zeros(self._original_R.shape[1:]))
            self._Z = self(R[None])

            self._original_vars = self._original_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            # here the original graph is copied an the featuremap is replaced
            imported_vars = tf.import_graph_def(
                self._original_graph_def,
                input_map={feature_name: self._Z},
                return_elements=self._original_output_names + [v.name for v in self._original_vars])

            self._outputs = imported_vars[:len(self._original_output_names)]
            self._imported_vars = imported_vars[1:]
            self._session.run(tf.global_variables_initializer())

            var_values = self._original_session.run(self._original_vars)
            assigns = [tf.assign(imported_var, val)
                       for imported_var, val in zip(self._imported_vars, var_values)]
            self._session.run(assigns)

    @contextmanager
    def copied_session_and_graph_as_default(self):
        with self._session.as_default(), self._graph.as_default():
            yield

    def set_classification_loss(self):
        logits = self._outputs[0]
        with self.copied_session_and_graph_as_default():
            super().set_classification_loss(logits)

    def feature_map_shape(self):
        return self._original_R.shape

    def get_feature(self, feed_dict):
        return self._original_session.run(self._original_R, feed_dict=feed_dict)

    def analyze(self,
                feature_feed_dict,
                copy_feed_dict,
                batch_size=10,
                steps=10,
                beta=10,
                learning_rate=1,
                min_std=0.01,
                smooth_std=1,
                normalize_beta=True,
                session=None,
                pass_mask=None,
                progbar=False):

        if not hasattr(self, '_optimizer'):
            self._build_optimizer()

        feature = self.get_feature(feature_feed_dict)

        with self.copied_session_and_graph_as_default():
            return super()._analyze_featuremap(
                feature, copy_feed_dict, batch_size=batch_size, steps=steps,
                beta=beta, learning_rate=learning_rate, min_std=min_std,
                smooth_std=smooth_std, normalize_beta=normalize_beta,
                session=self._session, pass_mask=pass_mask, progbar=progbar)

    def predict(self, feature_feed_dict):
        feature = self._original_session.run(self._original_R,
                                             feed_dict=feature_feed_dict)

        with self.copied_session_and_graph_as_default():
            self._session.run([tf.assign(self._restrict_flow, False),
                               tf.assign(self._use_layer_input, False)])
            return self._session.run(self._outputs, {self._Z: feature})

    def state_dict(self):
        state = super().state_dict()
        state['feature_name'] = self._feature_name
        state['output_names'] = self._output_names
        return state


class _InnvestigateAPI:
    def __init__(self, model,
                 neuron_selection_mode="max_activation",
                 disable_model_checks=False,
                 **kwargs):
        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode
        self._disable_model_checks = disable_model_checks
        self._model = model

    def fit(self, X=None, batch_size=32, **kwargs):
        generator = _BatchSequence(X, batch_size)
        if 'verbose' in kwargs:
            self._fit_generator(generator, **kwargs)
        else:
            self._fit_generator(generator, verbose=0, **kwargs)

    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, shuffle=True):
        pass

    def analyze(self, X):
        raise NotImplementedError()

    def _get_state(self):
        return {
            "model_json": self._model.to_json(),
            "model_weights": self._model.get_weights(),
            "neuron_selection_mode": self._neuron_selection_mode,
            "disable_model_checks": self._disable_model_checks,
        }

    def save(self):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load`
        to resemble the analyzer.
        :return: The class name and the state.
        """
        state = self._get_state()
        class_name = self.__class__.__name__
        return class_name, state

    def save_npz(self, fname):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load_npz`
        to resemble the analyzer.
        :param fname: The file's name.
        """
        class_name, state = self.save()
        np.savez(fname, **{"class_name": class_name, "state": state})

    @classmethod
    def _state_to_kwargs(clazz, state):
        model_json = state.pop("model_json")
        model_weights = state.pop("model_weights")
        disable_model_checks = state.pop("disable_model_checks")
        neuron_selection_mode = state.pop("neuron_selection_mode")
        assert len(state) == 0

        model = keras.models.model_from_json(model_json)
        model.set_weights(model_weights)
        return {
            "model": model,
            "disable_model_checks": disable_model_checks,
            "neuron_selection_mode": neuron_selection_mode,
        }

    @staticmethod
    def load(clazz, state):
        """
        Resembles an analyzer from the state created by
        :func:`analyzer.save()`.
        :param clazz: The analyzer's class.
        :param state: The analyzer's state.
        """
        kwargs = clazz._state_to_kwargs(state)
        return clazz(**kwargs)

    @staticmethod
    def load_npz(fname):
        """
        Resembles an analyzer from the file created by
        :func:`analyzer.save_npz()`.
        :param fname: The file's name.
        """
        f = np.load(fname, allow_pickle=True)
        class_name = f["class_name"].item()
        cls = globals()[class_name]
        state = f["state"].item()
        return cls.load(cls, state)


class IBACopyGraphInnvestigate(IBACopyGraph, _InnvestigateAPI):
    def __init__(self, model,
                 neuron_selection_mode='max_activation',
                 feature_name=None,
                 estimator=None, feature_mean_std=None,
                 session=None,
                 copy_session_config=None,
                 disable_model_checks=False,):
        if contains_activation(model.layers[-1], 'softmax'):
            raise ValueError("The model should not contain a softmax activation. "
                             "Please use the model_wo_softmax function!")
        output_names = [model.output.name]
        graph = model.input.graph
        IBACopyGraph.__init__(self, feature_name, output_names, estimator, feature_mean_std,
                              graph, session, copy_session_config)
        _InnvestigateAPI.__init__(self, model, neuron_selection_mode)


    def analyze(self, X, neuron_selection=None):
        if not hasattr(self, '_optimizer'):
            with self.copied_session_and_graph_as_default():
                IBALayer.set_classification_loss(self, self._outputs[0])

        if(neuron_selection is not None and
           self._neuron_selection_mode != "index"):
            raise ValueError("Only neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")
        if(neuron_selection is None and
           self._neuron_selection_mode == "index"):
            raise ValueError("neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")

        if self._neuron_selection_mode == "index":
            neuron_selection = np.asarray(neuron_selection).flatten()
            if neuron_selection.size == 1:
                neuron_selection = np.repeat(neuron_selection, len(X[0]))
        elif self._neuron_selection_mode == 'max_activation':
            logits = self.predict({self._model.input: X})
            neuron_selection = np.argmax(logits, axis=1)

        capacity = []
        for x, target in zip(X, neuron_selection):
            capacity.append(super().analyze(
                feature_feed_dict={self._model.input: X},
                copy_feed_dict={self.target: np.array([target])},
            ))
        return np.concatenate(capacity)

    def predict(self, feature_feed_dict):
        return super().predict(feature_feed_dict)[0]

    def _get_state(self):
        state = super()._get_state()
        copy_state = self.state_dict()
        del copy_state['output_names']
        state.update(copy_state)
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        estimator_state = state.pop("estimator")
        feature_mean = state.pop("feature_mean")
        feature_std = state.pop("feature_std")
        feature_name =  state.pop("feature_name")
        kwargs = super()._state_to_kwargs(state)
        kwargs['feature_name'] = feature_name

        if estimator_state is not None:
            estimator = TFWelfordEstimator("")
            estimator.load_state_dict(estimator_state)
            kwargs['estimator'] = estimator
        else:
            kwargs['feature_mean_std'] = (feature_mean, feature_std)
        return kwargs
