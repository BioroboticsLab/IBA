import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras


def kl_div(x, lambda_, mean_r, std_r):
    x_norm = (x - mean_r) / std_r
    var_z = (1 - lambda_) ** 2

    log_var_z = tf.log(var_z)

    mu_z = x_norm * lambda_
    #log_var_z = tf.clip_by_value(log_var_z, -10, 10)

    capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
    return capacity


def gaussian_kernel(size, std):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(0., std)

    vals = d.prob(tf.cast(tf.range(start=-size,limit=size + 1), tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def gaussian_blur(x, std=1.):
    #if std is not None and std > 0:
        # Construct static conv layer with gaussian kernel
    kernel_size = tf.cast((tf.round(2 * std)) * 2 + 1, tf.int32)  # Cover 2.5 stds in both directions


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


def pre_softmax_tensors(Xs, should_find_softmax=True):
    """Finds the tensors that were preceeding a potential softmax."""
    softmax_found = False

    Xs = iutils.to_list(Xs)
    ret = []
    for x in Xs:
        layer, node_index, tensor_index = x._keras_history
        if kchecks.contains_activation(layer, activation="softmax"):
            softmax_found = True
            if isinstance(layer, keras.layers.Activation):
                ret.append(layer.get_input_at(node_index))
            else:
                layer_wo_act = copy_layer_wo_activation(layer)
                ret.append(layer_wo_act(layer.get_input_at(node_index)))

    if should_find_softmax and not softmax_found:
        raise Exception("No softmax found.")

    return ret


def IBACopyGraph:
    """
    Inject an IBALayer into an existing model by partially copying the model.
    IBACopyGraph is very useful for pretrained models which you cannot alter
    otherwise.  The main drawback is that changes to the original graph will
    have no effect on the explanations.  Adding IBALayer directly to our model
    might also be more memory efficient.
    """
    def __init__(self, ):
        pass



def IBALayer(keras.layers.Layer):
    def build(self, input_shape):
        self._alpha = self.add_weight(
            shape=input_shape[1:],
            initializer=keras.initializers.Constant(value=5),
            trainable=True)

        self._mean_x = tf.get_variable(name='mean_r', trainable=False,  dtype=tf.float32,
                                      initializer=lambda: self.estim.mean().astype('float32'))
        self._std_r = tf.get_variable(name='std_r', trainable=False,  dtype=tf.float32,
                                     initializer=lambda: self.estim.std().astype('float32'))
        self._restrict_flow = tf.get_variable(name='restrict_flow', trainable=False,
                                              dtype=tf.bool, initializer=0)
        alpha_init = 5
        self._alpha = tf.get_variable(name='alpha', initializer=alpha_init*tf.ones(shape))

        # optimization placeholders
        self._batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self._learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self._beta = tf.placeholder(tf.float32, shape=[], name='beta')
        self._target = tf.placeholder(tf.int32, [1])

    def call(self, inputs):
        shape = [1,] + inputs.shape[1:].as_list()

        self.min_std = tf.placeholder(tf.float32, shape=[], name='min_std')
        self.smooth_std = tf.placeholder(tf.float32, shape=[], name='smooth_std')

        self.active_neurons = tf.get_variable('active_neurons', shape, trainable=False)
        self.fmap_var = tf.get_variable('feature_map', shape, trainable=False)
        r = tf.tile(self.fmap_var, [self.batch_size, 1, 1, 1])


        std_r_min = tf.maximum(self.std_r, self.min_std)
        std_r_min = tf.clip_by_value(std_r_min, np.exp(-10), np.exp(10))

        # std_too_low = tf.cast(self.std_x < self.min_std, tf.float32)
        # std_large_enough = tf.cast(self.std_x >= self.min_std, tf.float32)

        lambda_pre_blur = tf.sigmoid(_alpha)
        lambda_ = gaussian_blur(lambda_pre_blur, std=self.smooth_std)

        noise = tf.random.normal([self.batch_size,] + shape[1:])
        # ε ~ N(μ_r, σ_r)
        eps = std_r_min * noise + self.mean_r

        # z = λR + (1 - λ)ε
        z = lambda_ * r + (1 - lambda_) *.eps

        # save capacityies
        self._capacity = kl_div(r, lambda_, self.mean_r, std_r_min)
        self._capacity_in_bits = self.capacity / np.log(2)
        self._capacity_mean = tf.reduce_mean(self.capacity)

        # save tensors for report
        self._report('lambda_pre_blur', lambda_pre_blur)
        self._report('lambda', lambda_)
        self._report('eps', eps)
        self._report('alpha', self.alpha)
        self._report('capacity', self._capacity)
        self._report('capacity_in_bits', self._capacity_in_bits)
        self._report('capacity_mean', self._capacity_mean)
        self._report_once('featuremap', featuremap)
        self._report_once('featuremap_mean', self._mean_r)
        self._report('featuremap_std', self._std_r)

        return tf.cond(self._restrict_flow, z, x)

    def build_optimizer_from_logits(self, logits):
        target_one_hot = tf.one_hot(self.target, depth=self.logits.shape[-1])
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

    def build_optimizer(self, model_loss):
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss = model_loss  + self.beta * self._capacity_mean
        self._opt_step = self.opt.minimize(self.loss, var_list=[self.alpha])

    def fit():
        pass

    def _keep(self, name, tensor):
        assert name not in self._report_tensors
        self._report_tensors[name] = tensor

    def analyze_featuremap(self, featuremap, target=None, batch_size=10,
                steps=10, beta=10, learning_rate=100,
                min_std=0.01,
                smooth_std=1,
                norm_beta=True,
                progbar=False):

        sess = K.get_session()
        featuremap, = sess.run(
            [self.input],
            feed_dict={self.model.input: sample})

        self.fmap_value = fmap_value
        print(beta, fmap_value.shape)

        if type(target) == int:
            target = [target]
        with self.sess.as_default(), self.graph.as_default():
            self.sess.run([
                self.alpha.initializer,
                tf.variables_initializer(self.opt.variables()),
                tf.assign(self.fmap_var, featuremap),
                tf.assign(self.active_neurons, self.estim.active_neurons()),
            ])
            self.alphas.append(self.sess.run(self.alpha))

            self.sess.run(
                    [],
                    feed_dict={
                        self.batch_size: batch_size,
                        self.beta: beta,
                        self.min_std: min_std,
                        self.smooth_std: smooth_std,
                        self.target: target,
                        self.learning_rate: learning_rate,
                    })

            for step in trange(steps, disable=not progbar):
                opt_step = self.sess.run(
                    [self.opt_step],
                    feed_dict={
                        self.batch_size: batch_size,
                        self.beta: beta,
                        self.min_std: min_std,
                        self.smooth_std: smooth_std,
                        self.target: target,
                        self.learning_rate: learning_rate,
                    })

        self.sess.run(
            [self.logits, self.capacity, self.cross_entropy, self.loss],
            feed_dict={
                self.batch_size: batch_size,
                self.beta: beta,
                self.min_std: min_std,
                self.smooth_std: smooth_std,
                self.target: target,
                self.learning_rate: learning_rate,
            })

        return capacity

    def collect(*names):
        pass

    def report():
        pass
