import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import cv2
from contextlib import contextmanager


def resize(arr, shape, interp="bilinear"):
    # TODO: use skimage
    if interp == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp == "bilinear" or interp == "linear":
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError(interp)
    return cv2.resize(arr, dsize=shape, interpolation=interp)


def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()


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


class WelfordEstimator(Estimator):
    def __init__(self, channels, height, width, eps=1e-5):
        super().__init__()
        self.register_buffer('m', torch.zeros(channels, height, width))
        self.register_buffer('s', torch.zeros(channels, height, width))
        self.register_buffer('n_samples', torch.tensor([0], dtype=torch.long))

    def forward(self, x):
        for xi in x:
            old_m = self.m.clone()
            self.m = self.m + (xi-self.m) / (self.n_samples.float() + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self.n_samples += 1

    def mean(self):
        return self.m

    def std(self):
        return torch.sqrt(self.s / (self.n_samples.float() - 1))


class StopExecution(Exception):
    pass


class PerSampleBottleneck(nn.Module):
    """
    The Per Sample Bottleneck.  Is inserted in a existing model to suppress
    information, parametrized by a suppression mask alpha.  """
    def __init__(self,
                 channels,
                 height,
                 width,
                 sigma=1.,
                 beta=10,
                 steps=10,
                 lr=1,
                 batch_size=10,
                 estimator='welford',
                 initial_alpha=5.0,
                 reduction='bits-per-pixel',
                 progbar=False,
                 relu=False):
        """
        :param mean: The empirical mean of the activations of the layer
        :param std: The empirical standard deviation of the activations of the layer
        :param sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        :param relu: True if output should be clamped at 0, to imitate a post-ReLU distribution
        """
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width
        self.relu = relu
        self.initial_alpha = initial_alpha
        self.alpha = nn.Parameter(initial_alpha * torch.ones(channels, height, width))
        self.beta = beta
        self.batch_size = batch_size
        self.progbar = progbar
        self.lr = lr
        self.reduction = reduction
        self.train_steps = steps

        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None  # Filled on forward pass, used for loss

        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels)
        else:
            self.smooth = None

        if type(estimator) == str:
            estimator = {
                'welford': WelfordEstimator(channels, height, width),
                'batch-norm': WelfordEstimator(channels, height, width),
            }[estimator]
        self.estimator = estimator
        self.mean = None
        self.std = None
        self._restrict_information = False
        self._stop_execution = False

    def _reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)
        return self.alpha

    def forward(self, x):
        if self._restrict_information:
            return self._do_restrict_information(x)

        if self.training:
            self.estimator(x)
        if self._stop_execution:
            raise StopExecution()
        return x

    @contextmanager
    def stop_execution(self):
        self._stop_execution = True
        try:
            yield
        except StopExecution:
            self._stop_execution = False
        finally:
            self._stop_execution = False

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
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
        mu, log_var = x_norm * lamb, torch.log(1-lamb)

        # Sample new output values from p(z|x)
        z_norm = self._sample_z(mu, log_var)
        self.buffer_capacity = self._calc_capacity(mu, log_var)

        # Denormalize z to match magnitude of x
        z = z_norm * self.std + self.mean

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)
        return z

    def estimate(self, model, loader, device=None, progbar=False):
        """ Estimate mean and variance.  """

        try:
            import tqdm
        except ImportError:
            progbar = False
        if progbar:
            loader = tqdm.tqdm(loader)
        if device is None:
            device = next(iter(model.parameters())).device
        for (imgs, _) in loader:
            with torch.no_grad():
                with self.stop_execution():
                    model(imgs.to(device))

    @contextmanager
    def restrict_information(self):
        self._restrict_information = True
        try:
            yield
        finally:
            self._restrict_information = False

    def heatmap(self, input_t, model_loss_fn):
        assert input_t.shape[0] == 1, "We can only fit one sample a time"

        batch = input_t.expand(self.batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self._reset_alpha()
        optimizer = torch.optim.Adam(lr=self.lr, params=[self.alpha])
        self.mean = self.estimator.mean()
        self.std = self.estimator.std()

        with self.restrict_information():
            for _ in tqdm(range(self.train_steps), desc="Training Bottleneck",
                          disable=not self.progbar):
                optimizer.zero_grad()
                model_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling with 1/K
                information_loss = self.buffer_capacity.mean()
                loss = model_loss + self.beta * information_loss
                loss.backward()
                optimizer.step()

        return self._current_heatmap(input_t.shape[2:])

    def capacity(self):
        """
        returns the currenct capacity
        """
        return self.buffer_capacity[0]

    def _current_heatmap(self, shape=None):
        # Read bottleneck
        heatmap = to_np(self.buffer_capacity[0])
        if self.reduction == 'bits-per-pixel':
            heatmap = heatmap.sum(0) / float(np.log(2))
            print(heatmap.shape)
            if shape is not None:
                ho, wo = heatmap.shape
                h, w = shape
                # scale bit to the pixels
                heatmap *= (ho*wo) / (h*w)
                return resize(heatmap, shape)
        elif self.reduction is None or self.reduction == 'none':
            return heatmap
        elif self.reduction == 'sum':
            return heatmap.sum(0)
        else:
            raise


def insert_into_sequential(sequential, layer, idx):
    children = list(sequential.children())
    children.insert(idx, layer)
    return nn.Sequential(*children)
