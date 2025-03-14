import torch
from ccai.kernels import RBFKernel
from torch.distributions import MultivariateNormal
import numpy as np


class GPSurfaceModel:

    def __init__(self, train_x, train_y, lengthscale=1.5, outputscale=0.5):
        self.train_x = train_x
        self.train_y = train_y.flatten()
        self.kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)
        self.sigma_sq_noise = 0.01
        self.training = True

    def prior(self):
        K, grad_K, hess_K = self.kernel(self.train_x, self.train_x)
        return MultivariateNormal(torch.zeros_like(self.train_y), covariance_matrix=K)

    def posterior(self, x):
        # Returns joint distribution of query points x
        mu, _, _ = self.posterior_mean(x)

        # posterior covariance

        train_K, _, _ = self.kernel(self.train_x, self.train_x)
        K, _, _ = self.kernel(x, self.train_x)
        test_K, _, _ = self.kernel(x, x)

        eye = torch.eye(len(self.train_x), device=x.device)

        tmp = torch.linalg.solve(train_K + self.sigma_sq_noise * eye, K.permute(1, 0))
        innovation = K @ tmp
        cov = test_K - innovation

        return MultivariateNormal(mu, covariance_matrix=cov)

    def posterior_mean(self, x):
        """
            Return posterior mean with gradient and hessian
        :param x: Input tensor
        :return:
            y: output tensor - mean function evaluated at x
            grad_y: derivative of mean function wrt x
            hess_y: hessian of mean function wrt x
        """

        train_K, _, _ = self.kernel(self.train_x, self.train_x)
        K, grad_K, hess_K = self.kernel(x, self.train_x)
        eye = torch.eye(len(self.train_x), device=x.device)

        # compute [K + sigma_sq I]^-1 y
        tmp = torch.linalg.solve(train_K + self.sigma_sq_noise * eye, self.train_y.reshape(-1, 1))
        y = K @ tmp
        grad_y = grad_K.permute(0, 2, 1) @ tmp
        hess_y = hess_K.permute(0, 2, 3, 1) @ tmp

        return y.squeeze(-1), grad_y.squeeze(-1), hess_y.squeeze(-1)


class BatchGPSurfaceModel:
    def __init__(self):
        from functorch import vmap

        self.kernel = RBFKernel(lengthscale=1.5, outputscale=0.5)
        self.sigma_sq_noise = 0.01
        self.training = True

    def posterior_mean(self, x, train_x, train_y):
        """
            Return posterior mean with gradient and hessian
        :param x: (n_test, 2) Input tensor
        :param train_x: (n_train, 2) Training input tensor
        :param train_y: (n_train, 1) Training output tensor

        :return:
            y: output tensor - mean function evaluated at x
            grad_y: derivative of mean function wrt x
            hess_y: hessian of mean function wrt x
        """

        train_K, _, _ = self.kernel(train_x, train_x)
        K, grad_K, hess_K = self.kernel(x, train_x)

        N, _ = train_x.shape

        eye = torch.eye(N, device=x.device)

        # compute [K + sigma_sq I]^-1 y
        tmp = torch.linalg.solve(train_K + self.sigma_sq_noise * eye, train_y.reshape(N, 1))  # (n_train, 1)
        y = K @ tmp  # ( n_test, 1)
        grad_y = grad_K.permute(0, 2, 1) @ tmp
        hess_y = hess_K.permute(0, 2, 3, 1) @ tmp

        return y.squeeze(-1), grad_y.squeeze(-1), hess_y.squeeze(-1)


def get_random_surface():
    N = 10
    xs = torch.linspace(-5, 5, steps=N)
    ys = torch.linspace(-5, 5, steps=N)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    train_x = torch.stack((x.flatten(), y.flatten()), dim=1)
    train_y = torch.randn(len(train_x), 1)
    initial_gp_model = GPSurfaceModel(train_x, train_y)

    # sample from GP prior to get our data
    samples = initial_gp_model.prior().sample()
    return GPSurfaceModel(train_x, samples)


def get_random_obs(start, goal):
    N = 10
    lengthscale = 1
    outputscale = 1
    # we will say that start and goal are very far from an obstacle, but that is all
    # randomly create passage
    passage_x = 6 * np.random.rand(1) - 3
    passage_y = - passage_x
    passage = np.concatenate([passage_x, passage_y])
    train_x = torch.tensor(np.stack((start[:2], goal[:2], passage), axis=0)).to(dtype=torch.float)
    train_y = torch.tensor([[-1.], [-1.], [-2.]])
    initial_gp_model = GPSurfaceModel(train_x, train_y, lengthscale=lengthscale, outputscale=outputscale)

    # get a sample from the resulting posterior
    xs = torch.linspace(-5, 5, steps=N)
    ys = torch.linspace(-5, 5, steps=N)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    train_x = torch.stack((x.flatten(), y.flatten()), dim=1)
    samples = initial_gp_model.posterior(train_x).sample()
    # want to make environment slightly clearer so that it is more traversable on average
    samples = samples - 0.5
    return GPSurfaceModel(train_x, samples, lengthscale=lengthscale, outputscale=outputscale)
