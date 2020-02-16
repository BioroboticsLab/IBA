import numpy as np
import torch

from collections import OrderedDict


class WelfordEstimator:
    """
    Estimates the mean and standard derivation.
    For the algorithm see ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``.
    Args:
        channels: The number of channels of the feature map
        height: The heigth of the feature map
        width: The width of the feature map
    Example:
        Given a batch of images ``imgs`` with shape ``(10, 3, 64, 64)``, the mean and std could
        be estimated as follows::
            imgs = torch.randn(10, 3, 64, 64)
            estim = WelfordEstimator(3, 64, 64)
            estim(imgs)
            # returns the estimated mean
            estim.mean()
            # returns the estimated std
            estim.std()
            # returns the number of samples, here 10
            estim.n_samples()
            # returns a mask with active neurons
            estim.active_neurons()
    """
    def __init__(self, height, width, channels):
        super().__init__()
        shape = (1, height, width, channels)
        self.m = np.zeros(shape)
        self.s = np.zeros(shape)
        self._n_samples = 0
        self._neuron_nonzero = np.zeros(shape, dtype='long')

    def fit(self, x):
        """ Update estimates without altering x """
        for xi in x:
            self._neuron_nonzero += (xi != 0.)
            old_m = self.m.copy()
            self.m = self.m + (xi-self.m) / (self._n_samples + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return int(self._n_samples.item())

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """returns the estimate of the standard derivation."""
        return np.sqrt(self.s / (self._n_samples - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.astype(np.float32) / self._n_samples) > threshold


def get_output_shapes(model, input_t, layer_type=None):
    """Returns a dictionary from ``module_nam`` to output shape. Helpful to figure out the
    shape of the bottleneck."""
    if layer_type is None:
        layer_type = object
    sizes = OrderedDict()

    def save_output_shape(name):
        def wrapper(m, ins, outs):
            sizes[name] = outs[0].shape
        return wrapper

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, layer_type):
            hooks.append(layer.register_forward_hook(save_output_shape(name)))

    _ = model(input_t)

    for hook in hooks:
        hook.remove()
    return sizes


def _set_absmax(hmap, absmax=1.0):
    """ set the maximal amplitude below or above zero """
    current_absmax = max(abs(hmap.max()), abs(hmap.min()))
    if current_absmax:  # dont divide by 0
        scale = 1 / current_absmax * absmax
        return hmap * scale
    else:
        print("WARNING current_absmax == 0! mean {}, max {}, min {}, std {}", hmap.mean(), hmap.max(), hmap.min())
        return hmap


def _to_rgb(img):
    """Converts a gray scale image to an RGB image """
    if len(img.shape) == 2:
        return np.stack((img, img, img), axis=2)
    elif img.shape[2] == 1:
        return np.dstack((img, img, img))
    else:
        # nothing to do
        return img


def denormalize(img, method='imagenet'):
    """
    Reverses the preprocessing for imagenet.
    """
    if type(img) == torch.Tensor:
        if len(img.shape) == 3:
            img = img.cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            elif img.shape[0] == 1:
                img = img[0]
            else:
                raise ValueError('torch images must have 1 or 3 channels. Got {}'
                                 .format(img.shape[0]))
        elif len(img.shape) == 2:
            img = img.cpu().numpy()
        else:
            raise ValueError('torch images must have 2 or 3 dimensions. Got shape {}'
                             .format(img.shape))
    assert method == 'imagenet'
    mean3 = [0.485, 0.456, 0.406]
    std3 = [0.229, 0.224, 0.225]
    mean1 = [0.5]
    std1 = [0.5]
    mean, std = (mean3, std3) if img.shape[2] == 3 else (mean1, std1)
    for d in range(len(mean)):
        img[:, :, d] += mean[d]
        if np.max(img) > 1:
            img = img / np.max(img)  # force max 1
    return img


def _to_np_img(img: torch.Tensor, denorm=False):
    """

    """
    # force 2-3 dims
    if len(img.shape) == 4:
        img = img[0]
    # tensor to np
    if isinstance(img, torch.Tensor):
        img = img.detach()
        if img.is_cuda:
            img = img.cpu()
        img = img.numpy()
    # if color is not last
    if len(img.shape) > 2 and img.shape[0] < img.shape[2]:
        img = np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)
    if denorm:
        img = denormalize(img)
    return img


def plot_heatmap(heatmap, img=None, ax=None, label='Bits / Pixel',
                 min_alpha=0.2, max_alpha=0.7, vmax=None,
                 colorbar_size=0.3, colorbar_pad=0.08):

    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the image.

    Args:
        heatmap: the heatmap
        img: show this image
        ax: matplotlib axis. If ``None``, a new plot is created
        label: label for the colorbar
        min_alpha: minimum alpha value for the overlay. only used if ``img`` is given
        max_alpha: maximum alpha value for the overlay. only used if ``img`` is given
        vmax: maximum value for colorbar
        colorbar_size: width of the colorbar. default: Fixed(0.3)
        colorbar_pad: width of the colorbar. default: Fixed(0.08)

    Returns:
        The matplotlib axis ``ax``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    if img is not None:
        ax.imshow(_to_rgb(denormalize(_to_np_img(img)).mean(2)))

    ax1_divider = make_axes_locatable(ax)
    if type(colorbar_size) == float:
        colorbar_size = Fixed(colorbar_size)
    if type(colorbar_pad) == float:
        colorbar_pad = Fixed(colorbar_pad)
    cax1 = ax1_divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    if vmax is None:
        vmax = heatmap.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = 256
    half_jet_rgba = plt.cm.jet(np.linspace(0.5, 1, n))
    half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
    cmap = mpl.colors.ListedColormap(half_jet_rgba)
    hmap_jet = cmap(norm(heatmap))
    if img is not None:
        hmap_jet[:, :, -1] = (max_alpha - min_alpha)*norm(heatmap) + min_alpha
    ax.imshow(hmap_jet, alpha=1.)
    cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm)
    cbar.set_label(label, fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    ax.set_frame_on(False)
    return ax
