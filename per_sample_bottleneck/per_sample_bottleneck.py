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
    """
    children = list(sequential.children())
    children.insert(idx, layer)
    return nn.Sequential(*children)


def patch_layer(layer, btln):
    return nn.Sequential([
        layer,
        btln,
    ])


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
        return []

    def forward(self, x):
        return self.conv(self.pad(x))


class Estimator(nn.Module):
    def mean(self):
        raise NotImplementedError()

    def std(self):
        raise NotImplementedError()

    def n_samples(self):
        raise NotImplementedError()


class WelfordEstimator(Estimator):
    def __init__(self, channels, height, width, eps=1e-5):
        super().__init__()
        self.register_buffer('m', torch.zeros(channels, height, width))
        self.register_buffer('s', torch.zeros(channels, height, width))
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def forward(self, x):
        for xi in x:
            old_m = self.m.clone()
            self.m = self.m + (xi-self.m) / (self._n_samples.float() + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1

    def n_samples(self):
        return int(self._n_samples.item())

    def mean(self):
        return self.m

    def std(self):
        return torch.sqrt(self.s / (self._n_samples.float() - 1))


class RunningEstimator(Estimator):
    def __init__(self, channels, height, width, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(channels*height*width, affine=False)
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def forward(self, x):
        self.bn(x.view(len(x), -1))
        self._n_samples += len(x)

    def n_samples(self):
        return int(self._n_samples.item())

    def mean(self):
        return self.bn.running_mean.view(self.channels, self.height, self.width)

    def std(self):
        return self.bn.running_var.view(self.channels, self.height, self.width).sqrt()


class InterruptExecution(Exception):
    pass


class PerSampleBottleneck(nn.Module):
    """
    The Per Sample Bottleneck provides attribution heatmaps for you model.

    To apply it to your model you have to:
        1. Add it to your model. Either by including it as a module or by inserting it into a
        existing one with ``insert_into_sequential`` or ``insert_after_layer``.

        2. Estimate the mean and variance by calling ``estimate(model, datagen)``.

        3. Call ``.heatmap()``


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
        estimate_during_training: estimate mean and std during training
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
                 estimate_during_training=False,
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
        self.estimate_during_training = estimate_during_training
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None  # Filled on forward pass, used for loss
        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels)
        else:
            self.smooth = None
        self._use_welford = False
        self.estim_welford = WelfordEstimator(channels, height, width)
        self.estim_running = RunningEstimator(channels, height, width)
        self.mean = None
        self.std = None
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
        if self._use_welford:
            self.estim_welford(x)
        if self.training and self.estimate_during_training:
            self.estim_running(x)
        if self._interrupt_execution:
            raise InterruptExecution()
        return x

    @contextmanager
    def interrupt_execution(self):
        """
        Interrupts the execution of the model, once PerSampleBottleneck is called. Very useful
        for estimating.
        """
        self._interrupt_execution = True
        try:
            yield
        except InterruptExecution:
            pass
        finally:
            self._interrupt_execution = False

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ return mu with additive noise """
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    def _do_restrict_information(self, x):
        # Smoothen and expand a on batch dimension
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # Normalize x
        x_norm = (x - self.mean) / self.std

        # Get sampling parameters

        var = (1 - lamb) ** 2
        log_var = torch.log(var)
        mu = x_norm * lamb

        # Sample new output values from p(z|x)
        log_var = torch.clamp(log_var, -10, 10)
        z_norm = self._sample_z(mu, log_var)
        self.buffer_capacity = self._calc_capacity(mu, log_var)

        # Denormalize z to match magnitude of x
        z = z_norm * self.std + self.mean

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)
        return z

    @contextmanager
    def use_welford(self):
        self._use_welford = True
        try:
            yield
        finally:
            self._use_welford = False

    def estimate(self, model, loader, device=None, progbar=False):
        """ Estimate mean and variance using the welford estimator.
            Usually, using a 10.000 i.i.d. samples should be more than good enough.
        """
        try:
            import tqdm
        except ImportError:
            progbar = False
        if progbar == 'notebook':
            loader = tqdm.tqdm_notebook(loader)
        elif progbar:
            loader = tqdm.tqdm(loader)
        if device is None:
            device = next(iter(model.parameters())).device
        for (imgs, _) in loader:

            with torch.no_grad(), self.interrupt_execution(), self.use_welford():
                model(imgs.to(device))

    @contextmanager
    def supress_information(self):
        self._supress_information = True
        try:
            yield
        finally:
            self._supress_information = False

    def heatmap(self, input_t, model_loss_fn,
                estimator='auto',
                beta=None, min_std=None, optimization_steps=None,
                lr=None, batch_size=None, initial_alpha=None):
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

        if estimator == 'welford' or (estimator == 'auto' and self.estim_welford.n_samples() > 0):
            estim = self.estim_welford
        elif estimator in ['running', 'auto']:
            estim = self.estim_running
        else:
            raise ValueError("estimator must be either auto, welford, running. Got " + estimator)
        if estim.n_samples() < 1000:
            warnings.warn("Selected estimator was only fitted on {} samples. Might not be enough! We recommend 10.000 samples."
                          .format(estim.n_samples()))
        self.mean = estim.mean()

        std = estim.std()
        self.std = torch.max(std, self.min_std*torch.ones_like(std))

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

        return self._current_heatmap(input_t.shape[2:])

    def capacity(self):
        """
        returns the currenct capacity from the last input
        """
        return self.buffer_capacity[0]

    def _current_heatmap(self, shape=None):
        # Read bottleneck
        heatmap = _to_np(self.buffer_capacity[0])
        heatmap = np.nansum(heatmap, 0) / float(np.log(2))
        if shape is not None:
            ho, wo = heatmap.shape
            h, w = shape
            # scale bit to the pixels
            heatmap *= (ho*wo) / (h*w)
            return resize(heatmap, shape, order=1, preserve_range=True)
