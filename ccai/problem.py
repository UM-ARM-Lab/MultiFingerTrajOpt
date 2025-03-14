import torch
import numpy as np
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute


class Problem(metaclass=ABCMeta):

    def __init__(self, start, goal, T, device, *args, **kwargs):
        self.start = start
        self.goal = goal
        self.T = T
        self.device = device

    @abstract_attribute
    def dx(self):
        pass

    @abstract_attribute
    def du(self):
        pass

    @abstract_attribute
    def dg(self):
        pass

    @abstract_attribute
    def dh(self):
        pass

    @abstract_attribute
    def x_max(self):
        pass

    @abstract_attribute
    def x_min(self):
        pass

    @abstractmethod
    def _preprocess(self, x):
        pass

    @abstractmethod
    def _objective(self, x):
        pass

    @abstractmethod
    def _con_eq(self, x, compute_grads=True, compute_hess=True):
        g, grad_g, hess_g = None, None, None
        return g, grad_g, hess_g

    @abstractmethod
    def _con_ineq(self, x, compute_grads=True, compute_hess=True):
        h, grad_h, hess_h = None, None, None
        return h, grad_h, hess_h

    @abstractmethod
    def get_initial_xu(self, N):
        pass

    @abstractmethod
    def update(self, start, goal, T):
        pass

    def get_cost(self, x):
        return self._objective(x)[0]


class ConstrainedSVGDProblem(Problem):
    slack_weight = 1

    @abstract_attribute
    def squared_slack(self):
        pass

    @abstract_attribute
    def dz(self):
        pass

    @abstractmethod
    def eval(self, x):
        pass

    def combined_constraints(self, augmented_x, compute_grads=True, compute_hess=True):
        N = augmented_x.shape[0]
        augmented_x = augmented_x.reshape(N, self.T, self.dx + self.du + self.dz)
        xu = augmented_x[:, :, :(self.dx + self.du)]
        z = augmented_x[:, :, -self.dz:]

        g, grad_g, hess_g = self._con_eq(xu, compute_grads=compute_grads, compute_hess=compute_hess)
        h, grad_h, hess_h = self._con_ineq(xu, compute_grads=compute_grads, compute_hess=compute_hess)

        # print(g.max(), g.min(), h.max(), h.min())

        if h is None:
            return g, grad_g, hess_g

        if self.squared_slack:
            h_aug = h + 0.5 * z.reshape(-1, self.dz * self.T) ** 2
        else:
            h_aug = h + self.slack_weight * z.reshape(-1, self.dz * self.T)

        if not compute_grads:
            if g is None:
                return h_aug, None, None
            return torch.cat((g, h_aug), dim=1), None, None

            # Gradients - gradient wrt z should be z
        if self.squared_slack:
            z_extended = torch.diag_embed(torch.diag_embed(z).permute(0, 2, 3, 1)
                                          ).permute(0, 3, 1, 4, 2)  # (N, T, dz, T dz)
        else:
            z_extended = self.slack_weight * torch.diag_embed(torch.diag_embed(torch.ones_like(z)).permute(0, 2, 3, 1)
                                                              ).permute(0, 3, 1, 4, 2)  # (N, T, dz, T dz)

        grad_h_aug = torch.cat((
            grad_h.reshape(N, self.T, self.dz, self.T, -1),
            z_extended), dim=-1)
        grad_h_aug = grad_h_aug.reshape(N, self.dh, self.T * (self.dx + self.du + self.dz))

        if compute_hess:
            # Hessians - second derivative wrt z should be identity
            # partial derivatives dx dz should be zero
            # eye is (N, T, dz, dz, dz)
            eye = torch.diag_embed(torch.eye(self.dz, device=self.device)).repeat(N, self.T, 1, 1, 1)
            # make eye (N, T, dz, T, dz, T, dz)
            eye = torch.diag_embed(torch.diag_embed(eye.permute(0, 2, 3, 4, 1)))
            eye = eye.permute(0, 4, 1, 5, 2, 6, 3)

            hess_h = hess_h.reshape(N, self.T, self.dz, self.T, self.dx + self.du, self.T, self.dx + self.du)
            hess_h_aug = torch.zeros(N, self.T, self.dz, self.T, self.dx + self.du + self.dz,
                                     self.T, self.dx + self.du + self.dz, device=self.device)
            hess_h_aug[:, :, :, :, :self.dx + self.du, :, :self.dx + self.du] = hess_h

            if self.squared_slack:
                hess_h_aug[:, :, :, :, -self.dz:, :, -self.dz:] = eye

            hess_h_aug = hess_h_aug.reshape(N, self.dh, self.T * (self.dx + self.du + self.dz),
                                            self.T * (self.dx + self.du + self.dz))

        else:
            hess_h_aug = None

        if g is None:
            return h_aug, grad_h_aug, hess_h_aug
        grad_g_aug = torch.cat((
            grad_g.reshape(N, self.dg, self.T, -1),
            torch.zeros(N, self.dg, self.T, self.dz, device=self.device)),
            dim=-1
        ).reshape(N, self.dg, self.T * (self.dx + self.du + self.dz))

        if compute_hess:
            hess_g_aug = torch.zeros(N, self.dg,
                                     self.T, (self.dx + self.du + self.dz),
                                     self.T, (self.dx + self.du + self.dz), device=self.device)
            hess_g_aug[:, :, :, :self.dx + self.du, :, :self.dx + self.du] = hess_g.reshape(N,
                                                                                            self.dg,
                                                                                            self.T,
                                                                                            self.dx + self.du,
                                                                                            self.T,
                                                                                            self.dx + self.du)
            hess_g_aug = hess_g_aug.reshape(N, self.dg,
                                            self.T * (self.dx + self.du + self.dz),
                                            self.T * (self.dx + self.du + self.dz))
            hess_c = torch.cat((hess_g_aug, hess_h_aug), dim=1)
        else:
            hess_c = None

        c = torch.cat((g, h_aug), dim=1)  # (N, dg + dh)
        grad_c = torch.cat((grad_g_aug, grad_h_aug), dim=1)

        return c, grad_c, hess_c

    def get_initial_z(self, x):
        self._preprocess(x)
        N = x.shape[0]
        h, _, _ = self._con_ineq(x, compute_grads=False, compute_hess=False)
        if h is not None:
            if self.squared_slack:
                # z = torch.where(h < 0, torch.sqrt(-2 * h), 0)
                z = torch.sqrt(2 * torch.abs(h))
            else:
                z = torch.where(h < 0, -h / self.slack_weight, 0)

            return z.reshape(-1, self.T, self.dz)


class IpoptProblem(Problem):
    def __init__(self, start, goal, T, *args, **kwargs):
        super().__init__(start, goal, T, device='cpu')

    def objective(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        J, _, _ = self._objective(tensor_x)
        return J.detach().reshape(-1).item()

    def objective_grad(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        _, grad_J, _ = self._objective(tensor_x)
        return grad_J.detach().numpy().reshape(-1)

    def objective_hess(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        dx = tensor_x.shape[0]
        _, _, hess_J = self._objective(tensor_x)
        return hess_J.reshape(self.T * (self.dx + self.du),
                              self.T * (self.dx + self.du)).detach().numpy()

    def obj_hvp(self, x, v):
        hess = self.objective_hess(x)
        return hess @ v

    def con_eq(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        g, _, _ = self._con_eq(tensor_x, compute_grads=False)
        if g is None:
            return None
        return g.reshape(-1).detach().numpy()

    def con_eq_grad(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        _, grad_g, _ = self._con_eq(tensor_x, compute_grads=True, compute_hess=False)
        if grad_g is None:
            return None
        return grad_g.reshape(self.dg, -1).detach().numpy()

    def con_eq_hess(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        _, _, hess_g = self._con_eq(tensor_x)
        if hess_g is None:
            return None
        return hess_g.reshape(self.dg, self.T * (self.dx + self.du), -1).numpy()

    def con_eq_hvp(self, x, v):
        hess = self.con_eq_hess(x)
        if hess is None:
            return None
        return np.sum(v.reshape(-1, 1, 1) * hess, axis=0)

    def get_bounds(self):
        prob_dim = self.T * (self.dx + self.du)
        ub = self.x_max.reshape(1, self.dx + self.du).repeat(self.T, 1).reshape(-1)
        lb = - ub
        return lb, ub

    def con_ineq(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        h, _, _ = self._con_ineq(tensor_x)
        if h is None:
            return None
        return -h.reshape(-1).detach().numpy()

    def con_ineq_grad(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        _, grad_h, _ = self._con_ineq(tensor_x, compute_grads=True, compute_hess=False)
        if grad_h is None:
            return None
        return -grad_h.reshape(self.dh, -1).detach().numpy()

    def con_ineq_hess(self, x):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        _, _, hess_h = self._con_ineq(tensor_x)
        if hess_h is None:
            return None
        return -hess_h.reshape(self.dh, self.T * (self.dx + self.du), -1).numpy()

    def con_ineq_hvp(self, x, v):
        hess = self.con_ineq_hess(x)
        if hess is None:
            return None
        return np.sum(v.reshape(-1, 1, 1) * hess, axis=0)


class NLOptProblem(Problem):
    def __init__(self, start, goal, T, *args, **kwargs):
        super().__init__(start, goal, T, device='cpu')

    def objective(self, x, grad):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        J, grad_J, _ = self._objective(tensor_x)
        if grad.size > 0:
            grad[:] = grad_J.reshape(-1).detach().numpy()
        return J.detach().reshape(-1).item()

    def con_eq(self, result, x, grad):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        g, grad_g, _ = self._con_eq(tensor_x, compute_grads=True, compute_hess=False)

        if grad.size > 0:
            grad[:] = grad_g.detach().numpy()

        result[:] = g.reshape(-1).detach().numpy()

    def con_ineq(self, result, x, grad):
        tensor_x = torch.from_numpy(x).reshape(1, self.T, -1).to(torch.float32)
        h, grad_h, _ = self._con_ineq(tensor_x, compute_grads=True, compute_hess=False)

        if grad.size > 0:
            grad[:] = grad_h.detach().numpy()

        result[:] = h.reshape(-1).detach().numpy()

    def get_bounds(self):
        prob_dim = self.T * (self.dx + self.du)
        ub = self.x_max.reshape(1, self.dx + self.du).repeat(self.T, 1).reshape(-1)
        lb = - ub
        return lb.numpy(), ub.numpy()


class UnconstrainedPenaltyProblem(Problem):

    @abstract_attribute
    def penalty(self):
        pass

    @abstractmethod
    def dynamics(self, x, u):
        pass

    def objective(self, x, compute_grads=False):
        J, _, _ = self._objective(x)
        g, _, _ = self._con_eq(x, compute_grads=compute_grads, compute_hess=False)
        h, _, _ = self._con_ineq(x, compute_grads=compute_grads, compute_hess=False)
        J = J.reshape(-1)
        if h is not None:
            J = J + self.penalty[0] * torch.sum(torch.clamp(h, min=0), dim=1)
        if g is not None:
            J = J + self.penalty[1] * torch.sum(g.abs(), dim=1)
        N, T, _ = x.shape
        J = J + torch.where(x > self.x_max.repeat(N, T, 1).to(device=self.device), self.penalty[0] * torch.ones_like(x),
                            torch.zeros_like(x)).sum(dim=1).sum(dim=1)
        J = J + torch.where(x < self.x_min.repeat(N, T, 1).to(device=self.device), self.penalty[0] * torch.ones_like(x),
                            torch.zeros_like(x)).sum(dim=1).sum(dim=1)

        return J
