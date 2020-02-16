import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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
