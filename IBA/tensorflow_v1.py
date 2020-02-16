
from collections import OrderedDict
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow import keras
from IBA.utils import WelfordEstimator


tf.disable_v2_behavior()


def kl_div(x, lambda_, mean_r, std_r):
    x_norm = (x - mean_r) / std_r
    var_z = (1 - lambda_) ** 2

    log_var_z = tf.log(var_z)

    mu_z = x_norm * lambda_

    capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
    return capacity


def gaussian_kernel(size, std):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(0., std)
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
    x = tf.pad(x, [[0, 0], [kh, kh], [kh, kh], [0, 0]], "REFLECT")
    return tf.nn.depthwise_conv2d(
        x,
        kernel,
        strides=[1, 1, 1, 1],
        padding='VALID',
        name='blurring',
    )


class IBALayer(keras.layers.Layer):
    def __init__(self, estimator=None, batch_size=10, **kwargs):
        self._estimator = estimator
        self._collect_names = []
        self._batch_size = batch_size

        self._report_tensors = OrderedDict()
        self._report_tensors_first = OrderedDict()

        super().__init__(**kwargs)

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
                ret[name] = self._report_tensors[name]
        return ret

    def include_in_report(self, *names):
        for name in names:
            assert name in self._report_tensors or name in self._report_tensors_first, \
                "not tensor found with name {}! Try one of these: {}".format(
                    name, self.available_report_variables())
        self._collect_names = names

    def available_report_variables(self):
        return sorted([self._report_tensors.keys()] + [self._report_tensors_first.keys()])

    def report(self):
        return self._log

    def build(self, input_shape):
        shape = [1, ] + input_shape[1:]
        k = np.prod([int(s) for s in shape]).astype(np.float32)
        # optimization placeholders
        self._learning_rate = tf.get_variable('learning_rate', dtype=tf.float32, initializer=1.)
        self._beta = tf.get_variable('beta', dtype=tf.float32,
                                     initializer=np.array(10./k).astype(np.float32))
        self._target = tf.get_variable('iba_target', dtype=tf.int32, initializer=[1])

        # trained parameters
        alpha_init = 5
        self._alpha = tf.get_variable(name='alpha', initializer=alpha_init*tf.ones(shape))

        # feature map
        self._featuremap = tf.get_variable('featuremap', shape, trainable=False)

        # std normal noise
        self._std_normal = tf.random.normal([self._batch_size, ] + shape[1:])

        if self._estimator is None:
            self._estimator = WelfordEstimator(*shape[1:])

        # mean of feature map r
        self._mean_r = tf.get_variable(
            name='mean_r', trainable=False,  dtype=tf.float32,
            initializer=lambda: self._estimator.mean().astype('float32'))
        # std of feature map r
        self._std_r = tf.get_variable(
            name='std_r', trainable=False,  dtype=tf.float32,
            initializer=lambda: self._estimator.std().astype('float32'))

        # flag to restrict the flow
        self._restrict_flow = tf.get_variable(name='restrict_flow', trainable=False,
                                              dtype=tf.bool, initializer=False)

        self._use_layer_input = tf.get_variable(name='use_layer_input', trainable=False,
                                                dtype=tf.bool, initializer=False)

        # min standard derivation per neuron
        self._min_std_r = tf.get_variable('min_std_r', dtype=tf.float32, initializer=0.1)
        # kernel size for gaussian blur
        self._smooth_std = tf.get_variable('smooth_std', dtype=tf.float32, initializer=1.)

    def call(self, inputs):
        # shape = [1,] + inputs.shape[1:].as_list()
        # self.active_neurons = tf.get_variable('active_neurons', shape, trainable=False)

        featuremap = tf.cond(self._use_layer_input,
                             lambda: inputs,
                             lambda: self._featuremap)

        R = tf.tile(featuremap, [self._batch_size, 1, 1, 1])

        std_r_min = tf.maximum(self._std_r, self._min_std_r)
        std_r_min = tf.clip_by_value(std_r_min, np.exp(-10), np.exp(10))

        # std_too_low = tf.cast(self.std_x < self.min_std_r, tf.float32)
        # std_large_enough = tf.cast(self.std_x >= self.min_std_r, tf.float32)

        lambda_pre_blur = tf.sigmoid(self._alpha)
        λ = gaussian_blur(lambda_pre_blur, std=self._smooth_std)

        # ε ~ N(μ_r, σ_r)
        ε = std_r_min * self._std_normal + self._mean_r

        Z = λ * R + (1 - λ) * ε

        # save capacityies
        self._capacity = kl_div(R, λ, self._mean_r, std_r_min)
        self._capacity_in_bits = self._capacity / np.log(2)
        self._capacity_mean = tf.reduce_mean(self._capacity)

        # save tensors for report
        self._report('lambda_pre_blur', lambda_pre_blur)
        self._report('lambda',  λ)
        self._report('eps', ε)
        self._report('alpha', self._alpha)
        self._report('capacity', self._capacity)
        self._report('capacity_in_bits', self._capacity_in_bits)
        self._report('capacity_mean', self._capacity_mean)
        self._report_first('featuremap', featuremap)
        self._report_first('featuremap_mean', self._mean_r)
        self._report('featuremap_std', self._std_r)

        return tf.cond(self._restrict_flow, lambda: Z, lambda: inputs)

    def _build_optimizer_from_logits(self, logits):
        target_one_hot = tf.one_hot(self._target, depth=self.logits.shape[-1])
        loss_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_one_hot,
            logits=self.logits,
            name='cross_entropy'
        )
        loss_ce_mean = tf.reduce_mean(loss_ce)
        self._report('logits', logits)
        self._report('cross_entropy', loss_ce)
        self._report('cross_entropy_mean', loss_ce_mean)
        return self.build_optimizer(loss_ce_mean)

    def _build_optimizer(self, model_loss):
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        loss = model_loss + self._beta * self._capacity_mean
        self._optimizer_step = self._optimizer.minimize(loss, var_list=[self._alpha])

    def fit(self, feed_dict):
        feat_values, = self._session.run([self.inputs], feed_dict=feed_dict)
        self._estimator.fit(feat_values)

    def analyze(self, model, sample,
                target=None,
                batch_size=10,
                steps=10,
                beta=10,
                learning_rate=100,
                min_std=0.01,
                smooth_std=1,
                norm_beta=True,
                session=None,
                progbar=False):
        if session is None:
            session = keras.backend.get_session()
        featuremap = session.run(self.input, feed_dict={model.input: sample})

        return self.analyze_featuremap(featuremap,
                                       target=target,
                                       batch_size=batch_size,
                                       steps=steps,
                                       beta=beta,
                                       learning_rate=learning_rate,
                                       min_std=min_std,
                                       smooth_std=smooth_std,
                                       norm_beta=norm_beta,
                                       session=session,
                                       progbar=False)

    def analyze_featuremap(self, featuremap,
                           target,
                           batch_size=10,
                           steps=10,
                           beta=10,
                           learning_rate=100,
                           min_std=0.01,
                           smooth_std=1,
                           norm_beta=True,
                           session=None,
                           progbar=False):
        if session is None:
            session = keras.backend.get_session()

        self._log = OrderedDict()
        if type(target) == int:
            target = [target]
        session.run([
            self._alpha.initializer,
            tf.variables_initializer(self._optimizer.variables()),
            tf.assign(self._featuremap, featuremap),
            # tf.assign(self.active_neurons, self.estim.active_neurons()),
        ])

        feed_dict = {
            self._batch_size: batch_size,
            self._beta: beta,
            self._min_std_r: min_std,
            self._smooth_std: smooth_std,
            self._target: target,
            self._learning_rate: learning_rate,
        }

        report_tensors = self._get_report_tensors()
        report_tensors_first = self._get_report_tensors_first()

        self._log['init'] = session.run(report_tensors_first, feed_dict=feed_dict)
        for step in trange(steps, disable=not progbar):
            outs = session.run(
                [self._optimizer_step] + report_tensors,
                feed_dict=feed_dict)
            self._log[step] = outs

        self._log['final'] = session.run(report_tensor, feed_dict=feed_dict)
        return capacity


class IBACopyGraph:
    """
    Inject an IBALayer into an existing model by partially copying the model.
    IBACopyGraph is very useful for pretrained models which you cannot alter
    otherwise.  The main drawback is that changes to the original graph will
    have no effect on the explanations.  Adding IBALayer directly to our model
    might also be more memory efficient.
    """

    def __init__(self, input_name, featuremap_name, logit_name,
                 graph=None,
                 session=None,
                 ):
        self._actual_graph = graph or tf.get_default_graph()
        self._actual_session = session or K.get_session()

        self._actual_graph_def = actual_graph.as_graph_def()

        self._graph = tf.Graph()
        self._session = tf.Session(graph=self.graph)

        self._actual_input = actual_graph.get_tensor_by_name(input_name)
        self._actual_R = actual_graph.get_tensor_by_name(featuremap_name)

        with self._session.as_default(), self._graph.as_default():
            R = tf.get_variable('replaced_feature_map', dtype=tf.float32,
                                initializer=tf.zeros(self._actual_R.shape))
            self._iba_layer = IBALayer()
            Z = self._iba_layer(R)

            self._logits, = tf.import_graph_def(g, input_map={featuremap_name: Z},
                                               return_elements=[logit_name])

            name_to_init = []
            for n in self._graph.as_graph_def().node:
                if "init" in n.name:
                    name_to_init.append(n.name)
                    print(n.name)
            self._session.run([self._graph.get_operation_by_name(n) for n in name_to_init])
            self._session.run(tf.global_variables_initializer())

            variables = OrderedDict()
            for var in actual_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variables[var.name] = var
                print(var.name)

            var_values = actual_session.run(list(variables.values()))
            self.var_values = var_values
            variable_new = [self._graph.get_tensor_by_name('import/' + n) for n in variables.keys()]
            self._session.run([tf.assign(v_new, v_val) for v_new, v_val in zip(variable_new, var_values)])


    def fit(self, feed_dict, session=None):
        session = session or K.get_session()
        feat_values, = session.run([self._actual_R], feed_dict=feed_dict)
        self._iba_layer._estimator.fit(feat_values)

    def _build_optimizer(self):
        self._iba_layer._build_optimizer_from_logits(self._logits)

    def analyze(self, model, sample,
                target=None,
                batch_size=10,
                steps=10,
                beta=10,
                learning_rate=100,
                min_std=0.01,
                smooth_std=1,
                norm_beta=True,
                session=None,
                progbar=False):
        # TODO
        self._iba_layer.analyze()

    def include_in_report(self, *names):
        self._iba_layer.include_in_report(*names)

    def available_report_variables(self):
        return self._iba_layer.available_report_variables()

    def report(self):
        return self._iba_layer.report()
