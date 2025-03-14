import torch
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from functorch import vmap, jacrev, grad
import torch.distributions as dist
from torch import optim
from torch import nn


def get_gmm(means, weights, scale):
    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.Normal(loc=means.detach(), scale=scale), 1)
    return dist.mixture_same_family.MixtureSameFamily(mix, comp)


def d_log_qU(U_samples, U_mean, U_sigma):
    return 2 * (U_samples - U_mean) / U_sigma ** 2


class MoG:
    def __init__(self, means, sigma, weights=None, td="cpu"):
        if weights is None:
            weights = torch.ones(means.shape[0], device=td) / means.shape[0]
        self.means = means.clone().detach()
        mix_d = torch.distributions.Categorical(weights)

        comp_d = torch.distributions.Independent(
            torch.distributions.Normal(self.means, sigma * torch.ones(means.shape, device=td)), 2
        )
        self.mixture = torch.distributions.MixtureSameFamily(mix_d, comp_d)
        self.grad_log_prob = grad(self.log_prob)

    def sample(self, n=None):
        return self.mixture.sample((n,)) if n is not None else self.mixture.sample()

    def log_prob(self, x):
        return self.mixture.log_prob(x)


class SVMPC:

    def __init__(self, problem, params):
        """

            SVMPC
            optionally uses flow to represent posterior q(U) --
            samples maintained in U space. q(U) is mixture of Gaussians, in Flow space

        """
        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.H = self.problem.T
        self.fixed_H = params.get('receding_horizon', True)
        self.N = params.get('N', 64)
        self.M = params.get('M', 4)
        self.lr = params.get('step_size', 0.01)
        self.device = params.get('device', 'cuda:0')
        self.sigma = params.get('sigma', [1.0] * self.du)
        self.prior_sigma = params.get('prior_sigma', [1.0] * self.du)

        self.lambda_ = params.get('lambda', 1)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.online_iters = params.get('online_iters', 100)
        self.use_true_grad = params.get('use_grad', False)
        self.includes_x0 = params.get('include_x0', False)

        self.kernel = structured_rbf_kernel
        self.kernel_grad = jacrev(self.kernel, argnums=0)

        self.grad_cost = jacrev(self._combined_rollout_cost, argnums=1)
        self.warmed_up = False

        self.sigma = torch.tensor(self.sigma, device=self.device)
        self.prior_sigma = torch.tensor(self.prior_sigma, device=self.device)
        # sample initial actions
        self.U = self.sigma * torch.randn(self.M, self.H, self.du, device=self.device)
        self.U.requires_grad = True

        self.optim = None
        self.reset()

    def _combined_rollout_cost(self, x, u):
        pred_x = self._rollout_dynamics(x, u)
        return self._cost(pred_x, u, False)

    def _grad_cost(self, x, u):
        N, T, du = u.shape
        pred_x = self._rollout_dynamics(x, u)

        J, grad_J, _ = self.problem._objective(pred_x)
        g, grad_g, _ = self.problem._con_eq(pred_x, compute_grads=True, compute_hess=False)
        h, grad_h, _ = self.problem._con_ineq(pred_x, compute_grads=True, compute_hess=False)
        total_grad = grad_J.clone()

        total_grad += torch.sum(self.problem.penalty * 2 * g.unsqueeze(-1) * grad_g, dim=1)
        total_grad += torch.sum(self.problem.penalty * 2 * h.unsqueeze(-1) * torch.where(h.unsqueeze(-1) > -0.1,
                                                                                         grad_h, 0), dim=1)

        # need dynamics grad
        trajectory = torch.cat((pred_x, u), dim=-1)
        grad_xu = self.problem.grad_dynamics_constraint(trajectory)
        total_grad = total_grad.reshape(N, -1, 1)
        grad_dx_du = grad_xu.reshape(N, T, self.dx, T, (self.dx + self.du))[:, :, :self.dx, :, self.dx:]
        grad_dx_du = grad_dx_du.reshape(N, T * self.dx, T * self.du)

        total_grad = grad_dx_du.permute(0, 2, 1) @ total_grad
        return total_grad.reshape(N, T, self.du)

    def _cost(self, x, u, normalize=True):
        xu = torch.cat((x, u), dim=-1)
        J = self.problem.objective(xu)
        if normalize:
            return (J - J.min()) / J.max()
        return J

    def _rollout_dynamics(self, x0, u):
        N, H, du = u.shape
        assert H == self.H
        assert du == self.du

        x = [x0.reshape(1, self.dx).repeat(N, 1)]

        for t in range(self.H):
            x.append(self.problem.dynamics(x[-1], u[:, t]))

        if self.includes_x0:
            return torch.stack(x[:-1], dim=1)
        return torch.stack(x[1:], dim=1)

    def step(self, state, **kwargs):
        self.U = self.U.detach()
        self.U.requires_grad = True
        self.optim = torch.optim.Adam(params=[self.U], lr=self.lr)

        if self.fixed_H or (not self.warmed_up):
            new_T = None
        else:
            new_T = self.problem.T - 1
            self.H = new_T

        self.problem.update(state, T=new_T, **kwargs)

        if self.warmed_up:
            iterations = self.online_iters
        else:
            iterations = self.warmup_iters
            self.warmed_up = True
        # backup previous U incase solve fails
        prev_U = self.U.clone()
        #try:
        for _ in range(iterations):
            self.update_particles(state)
        ##except:
        # #   print('SVGD update failed, defaulting to previous trajectory')
        #    self.U = prev_U.clone()

        with torch.no_grad():
            pred_x = self._rollout_dynamics(state, self.U)
            # compute costs
            costs = self._cost(pred_x, self.U)
            if not self.use_true_grad:
                costs -= torch.min(costs)
                costs /= torch.max(costs)
            # Update weights
            weights = torch.softmax(-costs / self.lambda_, dim=0)
            U = self.U

        self.U = U[torch.argsort(weights, descending=True)[:self.M]]
        self.weights = weights[torch.argsort(weights, descending=True)[:self.M]]
        self.prior = MoG(weights=self.weights, means=self.U, sigma=self.prior_sigma, td=self.device)

        pred_X = self._rollout_dynamics(state, self.U)
        pred_traj = torch.cat((pred_X, self.U), dim=-1)

        self.shift()

        return pred_traj[0].detach(), pred_traj.detach()

    def update_particles(self, state):

        # get kernel and kernel gradient
        self.U.requires_grad = True
        Kxx = self.kernel(self.U, self.U)
        grad_K = self.kernel_grad(self.U, self.U)
        grad_K = grad_K.reshape(self.M, self.M, self.M, self.H, self.du)
        grad_K = torch.mean(torch.einsum('nmmti->nmti', grad_K), dim=0)
        #prior_ll = self.prior.log_prob(self.U)
        #grad_prior = torch.autograd.grad(prior_ll.sum(), self.U)[0]

        if self.use_true_grad:
            grad_lik = - self.grad_cost(state, self.U) / self.lambda_
            grad_lik = torch.einsum('nnti->nti', grad_lik)
        else:
            # first N actions from each mixture
            with torch.no_grad():
                noise = torch.randn(self.N, self.M, self.H, self.du, device=self.device)
                U_samples = self.sigma * noise + self.U.unsqueeze(0)

                # rollout trajectories
                pred_x = self._rollout_dynamics(state, U_samples.reshape(-1, self.H, self.du))
                costs = self._cost(pred_x, U_samples.reshape(-1, self.H, self.du)).reshape(self.N, self.M)

                costs -= torch.min(costs)
                costs /= torch.max(costs)
                # Evaluate cost of action samples - evaluate each action set N times
                weights = torch.softmax(-costs / self.lambda_, dim=0)

                grad_lik = d_log_qU(U_samples, self.U.unsqueeze(0), self.sigma)
                grad_lik = (weights.reshape(self.N, self.M, 1, 1) * grad_lik).sum(dim=0)

        phi = grad_K + torch.tensordot(Kxx, grad_lik, 1) / self.M

        self.U.grad = -phi
        torch.nn.utils.clip_grad_norm_(self.U, 10000)
        self.optim.step()
        self.optim.zero_grad()

        # clip to be in bounds
        self.U.data = torch.clamp(self.U.data,
                                  min=self.problem.x_min[-self.problem.du:],
                                  max=self.problem.x_max[-self.problem.du:]
                                  )

    def shift(self):
        if self.fixed_H:
            self.U = torch.roll(self.U, -1, dims=1)
            self.U[:, -1] = self.sigma * torch.randn(self.M, self.du, device=self.device)
        else:
            self.U = self.U[:, 1:]
        self.prior = MoG(weights=self.weights, means=self.U, sigma=self.prior_sigma, td=self.device)

    def reset(self):
        # sample initial actions
        self.U = self.sigma * torch.randn_like(self.U)
        self.U.requires_grad = True
        self.warmed_up = False
        self.weights = torch.ones(self.M, device=self.device) / self.M
        self.prior = MoG(weights=self.weights, means=torch.zeros_like(self.U), sigma=self.prior_sigma, td=self.device)
