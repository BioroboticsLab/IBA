import numpy as np

from collections import OrderedDict

# this module should be independent of torch and tensorflow
assert 'torch' not in globals()
assert 'tf' not in globals()
assert 'tensorflow' not in globals()

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
    def __init__(self):
        super().__init__()
        self.m = None
        self.s = None
        self._n_samples = 0
        self._neuron_nonzero = None

    def _init(self, shape):
        self.m = np.zeros(shape)
        self.s = np.zeros(shape)
        self._neuron_nonzero = np.zeros(shape, dtype='long')

    def fit(self, x):
        """ Update estimates without altering x """
        if self._n_samples == 0:
            # Initialize on first datapoint
            self._init(x.shape[1:])
        for xi in x:
            self._neuron_nonzero += (xi != 0.)
            old_m = self.m.copy()
            self.m = self.m + (xi-self.m) / (self._n_samples + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return self._n_samples

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


def plot_heatmap(heatmap, img=None, ax=None, label='Bits / Pixel',
                 min_alpha=0.2, max_alpha=0.7, vmax=None,
                 colorbar_size=0.3, colorbar_pad=0.08):

    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the image.

    Args:
        heatmap: np.ndarray the heatmap
        img: np.ndarray show this image under the heatmap
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
        # Underlay the image as greyscale
        grey = img.mean(2)
        ax.imshow(np.stack((grey, grey, grey), axis=2))

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
    half_jet_rgba = plt.cm.seismic(np.linspace(0.5, 1, n))
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
