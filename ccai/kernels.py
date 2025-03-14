import torch
import numpy as np

EPS = 1e-9


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    if len(tensor.shape) > 2:
        n = tensor.shape[1]
    else:
        n = tensor.shape[0]
        tensor = tensor.unsqueeze(0)
    m = tensor.shape[0]
    idx = torch.triu_indices(n, n, 1)
    tensor = tensor[:, idx[0], idx[1]]
    tensor = tensor.detach().reshape(m, -1)
    tensor_max = tensor.max(dim=1).values.reshape(-1, 1)
    return (torch.cat((tensor, tensor_max), dim=1).median(dim=1).values + tensor.median(dim=1).values) / 2.


def rbf_kernel(X, Xbar, Q=None):
    # X is N x d
    n = X.shape[0]
    X = X.reshape(n, -1)
    Xbar = Xbar.reshape(n, -1)
    n, d = X.shape
    diff = X.unsqueeze(0) - Xbar.unsqueeze(1)
    # Q must be PD - add diagonal to make it so
    if Q is not None:
        scaled_diff = diff @ Q.unsqueeze(0)
    else:
        scaled_diff = diff

    scaled_diff = (scaled_diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(n, n)
    h = median(torch.sqrt(scaled_diff)) ** 2
    h = h / np.log(n) + EPS
    # h = 0.1
    return torch.exp(-scaled_diff / h)


def get_chunk(X, num_chunks, expected_chunk_size):
    x = torch.chunk(X, chunks=num_chunks, dim=1)
    if x[-1].shape[1] < expected_chunk_size:
        x = x[:-1]
    return x


def structured_rbf_kernel(X, Xbar, Q=None):
    # X is N x T x d
    n, T, d = X.shape
    mod = T % 3
    num_chunks = T // 3
    if Q is not None:
        Xbar = (Q.reshape(1, T * d, T * d) @ Xbar.reshape(n, T * d, 1)).reshape(n, T, d)

    x1 = get_chunk(X, num_chunks, 3)
    x2 = get_chunk(X[:, 1:], num_chunks, 3)
    x3 = get_chunk(X[:, 2:], num_chunks, 3)
    x1bar = get_chunk(Xbar, num_chunks, 3)
    x2bar = get_chunk(Xbar[:, 1:], num_chunks, 3)
    x3bar = get_chunk(Xbar[:, 2:], num_chunks, 3)

    x = torch.stack((*x1, *x2, *x3), dim=0)
    xbar = torch.stack((*x1bar, *x2bar, *x3bar), dim=0)

    M = x.shape[0]
    x = x.reshape(M, n, -1)
    xbar = xbar.reshape(M, n, -1)
    d = x.shape[2]
    diff = x.unsqueeze(1) - xbar.unsqueeze(2)

    sq_diff = (diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(M, n, n)
    h = median(torch.sqrt(sq_diff)) ** 2
    h = h / np.log(n) + EPS
    # if Q is not None:
    #    h = d
    # h = 0.1
    return torch.exp(-sq_diff / h.reshape(M, 1, 1)).mean(dim=0)


class RBFKernel:

    def __init__(self,
                 #        use_median_trick=True,
                 lengthscale=None,
                 outputscale=None):
        # if not use_median_trick:
        #    if lengthscale is None or outputscale is None:
        #        raise ValueError('Must supply lengthscale and output scale if not using median heuristic')

        # self.use_median_trick = use_median_trick
        self.l = lengthscale
        self.sigma_sq = outputscale

    def __call__(self, X, Xbar):
        n, d = X.shape
        m = Xbar.shape[0]
        diff = X.unsqueeze(1) - Xbar.unsqueeze(0)  # diff should be n x m x d
        sq_diff = (diff.reshape(n, m, 1, d) @ diff.reshape(n, m, d, 1)).reshape(n, m)
        # if self.use_median_trick:
        #    h = torch.sqrt(median(sq_diff) / 2) + 1e-3
        #    sigma_sq = 1
        # else:
        h = self.l ** 2
        sigma_sq = self.sigma_sq

        K = sigma_sq * torch.exp(-0.5 * sq_diff / h)
        grad_K = - sigma_sq * diff * K.reshape(n, m, 1) / h
        hess_K = diff.reshape(n, m, d, 1) @ diff.reshape(n, m, 1, d) * K.reshape(n, m, 1, 1) / h ** 2
        hess_K = hess_K - torch.diag_embed(K.reshape(n, m, 1).expand(-1, -1, d)) / h
        hess_K = hess_K * sigma_sq
        return K, grad_K, hess_K
