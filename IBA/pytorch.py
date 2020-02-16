import numpy as np
import torch.nn as nn
import torch
import warnings
from tqdm import tqdm
from contextlib import contextmanager
from skimage.transform import resize


def _to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()


def insert_into_sequential(sequential, layer, idx):
    """
    Returns a ``nn.Sequential`` with ``layer`` inserted in ``sequential`` at position ``idx``.
    """
    children = list(sequential.children())
    children.insert(idx, layer)
    return nn.Sequential(*children)


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels,):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              padding=0, kernel_size=kernel_size,
                              groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def parameters(self):
        """returns no parameters"""
        return []

    def forward(self, x):
        return self.conv(self.pad(x))


class WelfordEstimator(nn.Module):
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
    def __init__(self, channels, height, width):
        super().__init__()
        self.register_buffer('m', torch.zeros(channels, height, width))
        self.register_buffer('s', torch.zeros(channels, height, width))
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))
        self.register_buffer('_neuron_nonzero', torch.zeros(channels, height, width,
                                                            dtype=torch.long))

    def forward(self, x):
        """ Update estimates without altering x """
        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            self.m = self.m + (xi-self.m) / (self._n_samples.float() + 1)
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
        return torch.sqrt(self.s / (self._n_samples.float() - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.float() / self._n_samples.float()) > threshold


class _InterruptExecution(Exception):
    pass


class PerSampleBottleneck(nn.Module):
    """
    The Per Sample Bottleneck provides attribution heatmaps for your model.

    Example:
        ::

            # Create the Per-Sample Bottleneck:
            btln = PerSampleBottleneck(channels, height, width)

            # Add it to your model
            model.conv4 = nn.Sequential(model.conv4, btln)

            # Estimate the mean and variance
            btln.estimate(model, datagen)

            # Create heatmaps
            model_loss_closure = lambda x: -torch.log_softmax(model(x), 1)[:, target].mean()
            heatmap = btln.heatmap(img[None].to(dev), model_loss_closure)
            ```

    Args:
        channels: C form an input of size `(N, C, H, W)`
        height: H form an input of size `(N, C, H, W)`
        width: W form an input of size `(N, C, H, W)`
        sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        beta: weighting of model loss and mean information loss.
        min_std: minimum std of the features
        lr: optimizer learning rate. default: 1
        batch_size: number of samples to use per iteration
        initial_alpha: initial value for the parameter.
    """
    def __init__(self,
                 channels,
                 height,
                 width,
                 sigma=1.,
                 beta=10,
                 min_std=0.01,
                 optimization_steps=10,
                 lr=1,
                 batch_size=10,
                 initial_alpha=5.0,
                 progbar=False,
                 relu=False):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.relu = relu
        self.beta = beta
        self.min_std = min_std
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.alpha = nn.Parameter(initial_alpha * torch.ones(channels, height, width))
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None  # Filled on forward pass, used for loss
        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels)
        else:
            self.smooth = None
        self.estimator = WelfordEstimator(channels, height, width)
        self._estimate = False
        self._mean = None
        self._std = None
        self._active_neurons = None
        self._supress_information = False
        self._interrupt_execution = False

    def _reset_alpha(self, initial_alpha=None):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(initial_alpha or self.initial_alpha)
        return self.alpha

    def forward(self, x):
        if self._supress_information:
            return self._do_restrict_information(x)
        if self._estimate:
            self.estimator(x)
        if self._interrupt_execution:
            raise _InterruptExecution()
        return x

    @contextmanager
    def interrupt_execution(self):
        """
        Interrupts the execution of the model, once PerSampleBottleneck is called. Useful
        for estimation when the model has only be executed until the Per-Sample Bottleneck.

        Example:
            Executes the model only until the bottleneck layer::

                with bltn.interrupt_execution():
                    out = model(x)
                    # out will not be defined
                    print("this will not be printed")
        """
        self._interrupt_execution = True
        try:
            yield
        except _InterruptExecution:
            pass
        finally:
            self._interrupt_execution = False

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ Return mu with additive noise """
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    def _do_restrict_information(self, x):
        """ Selectively remove information from x by applying noise """

        # Smoothen and expand a on batch dimension
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # Normalize x
        x_norm = (x - self._mean) / self._std

        # Get sampling parameters
        var = (1 - lamb) ** 2
        log_var = torch.log(var)
        mu = x_norm * lamb

        # Sample new output values from p(z|x)
        eps = mu.data.new(mu.size()).normal_()
        z_norm =  x_norm * lamb + (1-lamb) * eps
        self.buffer_capacity = self._calc_capacity(mu, log_var) * self._active_neurons

        # Denormalize z to match original magnitude of x
        z = z_norm * self._std + self._mean
        z *= self._active_neurons

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z

    @contextmanager
    def enable_estimation(self):
        """
        Context manager to enable estimation of the mean and standard derivation.
        We recommend to use the `self.estimate` method.
        """
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    def reset_estimate(self):
        """
        Resets the estimator. Useful if the distribution changes. Which can happen if you
        trained the model more.
        """
        self.estimator = WelfordEstimator(self.channels, self.height, self.width)

    def estimate(self, model, dataloader, device=None, n_samples=10000, progbar=False, reset=True):
        """ Estimate mean and variance using the welford estimator.
            Usually, using 10.000 i.i.d. samples gives decent estimates.

            Args:
                model: the model containing the bottleneck layer
                dataloader: yielding ``batch``'s where the first sample
                    ``batch[0]`` is the image batch.
                device: images will be transfered to the device. If ``None``, it uses the device
                    of the first model parameter.
                n_samples (int): run the estimate on that many samples
                progbar (bool): show a progress bar.
                reset (bool): reset the current estimate of the mean and std

        """
        try:
            import tqdm
        except ImportError:
            progbar = False
        if progbar == 'notebook':
            dataloader = tqdm.tqdm_notebook(dataloader)
        elif progbar:
            dataloader = tqdm.tqdm(dataloader)
        if device is None:
            device = next(iter(model.parameters())).device
        if reset:
            self.reset_estimate()
        self.estimator.to(device)
        for batch in dataloader:
            imgs = batch[0]
            if self.estimator.n_samples() > n_samples:
                break
            with torch.no_grad(), self.interrupt_execution(), self.enable_estimation():
                model(imgs.to(device))

    @contextmanager
    def supress_information(self):
        """
        Context mananger to enable information supression.

        Example:
            To make a prediction, with the information flow being supressed.::

                with btln.supress_information():
                    # now noise is added
                    model(x)
        """
        self._supress_information = True
        try:
            yield
        finally:
            self._supress_information = False

    def heatmap(self, input_t, model_loss_fn,
                estimator='auto',
                beta=None, min_std=None, optimization_steps=None,
                lr=None, batch_size=None, initial_alpha=None, active_neurons_threshold=0.01):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input image of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            estimator: either `auto`,`welford`, or `running`. `auto` picks
                `welford` if available otherwise `running`.
            beta: if not None, overrides the bottleneck beta value
            beta: if not None, overrides the bottleneck min_std value
            lr: if not None, overrides the bottleneck lr value
            batch_size: if not None, overrides the bottleneck batch_size value
            initial_alpha: if not None, overrides the bottleneck initial_alpha value
            active_neurons_threshold: used threshold to determine if a neuron is active

        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[0] == 1, "We can only fit one sample a time"

        beta = beta or self.beta
        min_std = min_std or self.min_std
        optimization_steps = optimization_steps or self.optimization_steps
        lr = lr or self.lr
        batch_size = batch_size or self.batch_size

        batch = input_t.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self._reset_alpha(initial_alpha)
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn("Selected estimator was only fitted on {} samples. "
                          "Might not be enough! We recommend 10.000 samples."
                          .format(self.estimator.n_samples()))
        self._mean = self.estimator.mean()
        std = self.estimator.std()
        self._active_neurons = self.estimator.active_neurons(active_neurons_threshold).float()
        self._std = torch.max(std, self.min_std*torch.ones_like(std))

        self._loss = []
        self._model_loss = []
        self._information_loss = []
        with self.supress_information():
            for _ in tqdm(range(optimization_steps), desc="Training Bottleneck",
                          disable=not self.progbar):
                optimizer.zero_grad()
                model_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling with 1/K
                information_loss = self.buffer_capacity.mean()
                loss = model_loss + beta * information_loss
                loss.backward()
                optimizer.step()

                self._loss.append(loss.item())
                self._model_loss.append(model_loss.item())
                self._information_loss.append(information_loss.item())

        return self._current_heatmap(input_t.shape[2:])

    def capacity(self):
        """ Returns a tensor with the currenct capacity from the last input.
        Shape is ``(self.channels, self.height, self.width)`` """
        return self.buffer_capacity[0]

    def _current_heatmap(self, shape=None):
        """ Return a 2D-heatmap of flowing information. Optionally resize the map to match a certain shape """
        heatmap = _to_np(self.buffer_capacity[0])
        heatmap = np.nansum(heatmap, 0) / float(np.log(2))
        if shape is not None:
            ho, wo = heatmap.shape
            h, w = shape
            # Scale bits to the pixels
            heatmap *= (ho*wo) / (h*w)
            return resize(heatmap, shape, order=1, preserve_range=True)
