import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from torch_cg import cg_batch


class ConstrainedSteinTrajOpt:

    def __init__(self, problem, params):
        self.dt = params.get('step_size', 1e-2)
        self.alpha_J = params.get('alpha_J', 1)
        self.alpha_C = params.get('alpha_C', 1)
        # self.momentum = params.get('momentum', 0)
        self.iters = params.get('iters', 100)
        self.penalty = params.get('penalty', 1e2)
        self.resample_sigma = params.get('resample_sigma', 1e-2)
        self.resample_temperature = params.get('resample_temperature', 0.1)
        self.use_momentum = params.get('use_momentum', False)
        if self.use_momentum:
            self.beta_1 = 0.5
            # self.beta_1 = 0.9
            # self.beta_2 = 0.999
            self.adam_eps = 1e-8
            self.first_moment = 0
            # self.second_moment = 0



        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.dg = problem.dg
        self.dh = problem.dh
        self.T = self.problem.T
        self.profile = profile

        self.matrix_kernel = False
        self.use_fisher = False
        self.use_hessian = False  # True
        self.use_constraint_hessian = False
        self.normxiJ = None
        #self.dtype = torch.float32
        self.dtype = torch.float64
        self.delta_x = None
        self.Bk = None

    def compute_update(self, xuz):
        N = xuz.shape[0]
        d = self.dx + self.du
        xuz = xuz.detach()
        xuz.requires_grad = True

        xuz = xuz.to(dtype=torch.float32)
        grad_J, hess_J, K, grad_K, C, dC, hess_C = self.problem.eval(xuz.to(dtype=torch.float32))
        if self.use_momentum:
            self.first_moment = self.beta_1 * self.first_moment + (1 - self.beta_1) * grad_J
            # self.second_moment = self.beta_2 * self.second_moment + (1 - self.beta_2) * grad_J ** 2
            # first_unbiased = self.first_moment / (1 - self.beta_1)
            # second_unbiased = self.second_moment / (1 - self.beta_2)
            # grad_J = first_unbiased / (torch.sqrt(second_unbiased) + self.adam_eps)
            grad_J = self.first_moment

            # implement rmsprop
            # self.first_moment = self.beta_1 * self.first_moment + (1 - self.beta_1) * grad_J ** 2
            # grad_J = grad_J / (torch.sqrt(self.first_moment) + self.adam_eps)



        if hess_C is None and self.use_constraint_hessian:
            hess_C = torch.zeros(N, self.dh + self.dg, self.T * d + self.dh, self.T * d + self.dh, device=xuz.device)

        with torch.no_grad():
            # we try and invert the dC dCT, if it is singular then we use the psuedo-inverse
            eye = torch.eye(self.dg + self.dh).repeat(N, 1, 1).to(device=C.device, dtype=self.dtype)
            dCdCT = dC @ dC.permute(0, 2, 1)
            dCdCT = dCdCT.to(dtype=self.dtype)
            A_bmm = lambda x: dCdCT @ x
            #
            # damping_factor = 1e-1
            try:
                damping_factor = 1e-6
                dCdCT_inv = torch.linalg.solve(dCdCT + damping_factor * eye, eye)
                # dCdCT_inv = torch.linalg.solve(dCdCT, eye)
                if torch.any(torch.isnan(dCdCT_inv)):
                    raise ValueError('nan in inverse')
            except Exception as e:
                print(e)
                # dCdCT_inv = torch.linalg.lstsq(dC @ dC.permute(0, 2, 1), eye).solution
                # dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1))
                dCdCT_inv, _ = cg_batch(A_bmm, eye, verbose=False)
            dCdCT_inv = dCdCT_inv.to(dtype=torch.float32)
            # get projection operator
            projection = dCdCT_inv @ dC
            eye = torch.eye(d * self.T + self.dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
            projection = eye - dC.permute(0, 2, 1) @ projection
            # compute term for repelling towards constraint
            xi_C = dCdCT_inv @ C.unsqueeze(-1)
            xi_C = (dC.permute(0, 2, 1) @ xi_C).squeeze(-1)
            # xi_C = (dC.permute(0, 2, 1) @ C.unsqueeze(-1)).squeeze(-1)

            # compute gradient for projection
            # now the second index (1) is the
            # x with which we are differentiating
            dCdCT_inv = dCdCT_inv.unsqueeze(1)

            dCT = dC.permute(0, 2, 1).unsqueeze(1)
            dC = dC.unsqueeze(1)

            if self.use_constraint_hessian:
                hess_C = hess_C.permute(0, 3, 1, 2)
                hess_CT = hess_C.permute(0, 1, 3, 2)
                # compute first term

                first_term = hess_CT @ (dCdCT_inv @ dC)
                second_term = dCT @ dCdCT_inv @ (hess_C @ dCT + dC @ hess_CT) @ dCdCT_inv @ dC
                third_term = dCT @ dCdCT_inv @ hess_C

                # add terms and permute so last dimension is the x which we are differentiating w.r.t
                grad_projection = (first_term - second_term + third_term).permute(0, 2, 3, 1)

                # compute total gradient of kernel
                # first term is grad of scalar
                grad_proj = torch.einsum('mijj->mij', grad_projection)

            # now we need to combine all the different projections together
            PP = projection.unsqueeze(0) @ projection.unsqueeze(1)  # should now be N x N x D x D

            if self.matrix_kernel:
                grad_K = torch.cat((grad_K, torch.zeros(N, N, self.T * d, self.dh, device=xuz.device)), dim=-1)
                grad_K = torch.cat((grad_K, torch.zeros(N, N, self.dh, self.T * d + self.dh, device=xuz.device)),
                                   dim=-2)

                first_term = projection.unsqueeze(0) @ grad_K @ projection.unsqueeze(1)
                K_extended = torch.eye(self.T * d + self.dh, self.T * d + self.dh, device=xuz.device).repeat(N, N, 1, 1)
                K_extended[:, :, :self.T * d, :self.T * d] = K
                first_term = torch.sum(first_term, dim=3)
                matrix_K = projection.unsqueeze(0) @ K_extended @ projection.unsqueeze(1)

            else:
                if hess_J is not None:
                    Q_inv = torch.eye(d * self.T + self.dh, device=xuz.device)
                    Q_inv = torch.linalg.inv(hess_J.unsqueeze(0))

                    PQ = projection @ Q_inv
                    PQP = PQ.unsqueeze(0) @ projection.unsqueeze(1)
                    first_term = torch.einsum('nmj, nmij->nmi', grad_K, PQP)
                    matrix_K = K.reshape(N, N, 1, 1) * PQ.unsqueeze(0) @ projection.unsqueeze(1)

                else:
                    PQ = projection
                    first_term = torch.einsum('nmj, nmij->nmi', grad_K, PP)
                    matrix_K = K.reshape(N, N, 1, 1) * projection.unsqueeze(0) @ projection.unsqueeze(1)

            grad_matrix_K = first_term
            if self.use_constraint_hessian:
                second_term = K.reshape(N, N, 1, 1) * PQ.unsqueeze(0) @ grad_proj.unsqueeze(1)
                second_term = torch.sum(second_term, dim=3)
                grad_matrix_K = grad_matrix_K + second_term

            grad_matrix_K = torch.sum(grad_matrix_K, dim=0)

            # compute kernelized score
            kernelized_score = torch.sum(matrix_K @ -grad_J.reshape(N, 1, -1, 1), dim=0)  # the projected gradient

            phi = self.gamma * kernelized_score.squeeze(-1) / N + grad_matrix_K / N  # maximize phi

            xi_J = -phi
            if False:
                # Normalize gradient
                normxiC = torch.clamp(torch.linalg.norm(xi_C, dim=1, keepdim=True, ord=np.inf), min=1e-9)
                # normxiC = torch.min(0.9 * torch.ones_like(normxiC) / self.dt, normxiC)
                xi_C = self.alpha_C * xi_C / normxiC

                # if self.normxiJ is None:
                #    self.normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                #    xi_J = self.alpha_J * xi_J / self.normxiJ
                # else:
                normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                xi_J = self.alpha_J * xi_J / normxiJ

        return (self.alpha_J * xi_J + self.alpha_C * xi_C).detach().to(dtype=torch.float32)

    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        # min_u = self.problem.u_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        # max_u = self.problem.u_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            if not self.problem.squared_slack:
                min_val = 0
            else:
                min_val = -1e3
            min_x = torch.cat((min_x, min_val * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)
            max_x = torch.cat((max_x, 1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

    def resample(self, xuz):
        N = xuz.shape[0]
        xuz = xuz.to(dtype=torch.float32)
        J = self.problem.get_cost(xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du])
        C, dC, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1),
                                                     compute_grads=True,
                                                     compute_hess=False)

        penalty = J.reshape(N) + self.penalty * torch.sum(C.reshape(N, -1).abs(), dim=1)
        # normalize penalty
        penalty = (penalty - penalty.min()) / (penalty.max() - penalty.min())
        # now we sort particles
        idx = torch.argsort(penalty, descending=False)
        xuz = xuz[idx]
        penalty = penalty[idx]
        weights = torch.softmax(-penalty / self.resample_temperature, dim=0, )
        weights = torch.cumsum(weights, dim=0)
        # now we resample
        p = torch.rand(N, device=xuz.device)
        idx = torch.searchsorted(weights, p)

        # compute projection for noise
        dCdCT = dC @ dC.permute(0, 2, 1)
        A_bmm = lambda x: dCdCT @ x
        eye = torch.eye(self.dg + self.dh).repeat(N, 1, 1).to(device=C.device)
        try:
            dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1), eye)
            if torch.any(torch.isnan(dCdCT_inv)):
                raise ValueError('nan in inverse')
        except Exception as e:
            print(e)
            # dCdCT_inv = torch.linalg.lstsq(dC @ dC.permute(0, 2, 1), eye).solution
            # dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1))
            dCdCT_inv, _ = cg_batch(A_bmm, eye, verbose=False)
        projection = dCdCT_inv @ dC
        eye = torch.eye((self.dx + self.du) * self.T + self.dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
        projection = eye - dC.permute(0, 2, 1) @ projection

        noise = self.resample_sigma * torch.randn_like(xuz)
        eps = projection[idx] @ noise.unsqueeze(-1)
        # print(penalty, weights)
        # we need to add a very small amount of noise just so that the particles are distinct - otherwise
        # they will never separate
        xuz = xuz[idx] + eps.squeeze(-1)

        return xuz

        # weights = weights / torch.sum(weights)

    def solve(self, x0, resample=False):
        self.normxiJ = None
        self.Bk = None
        # update from problem incase T has changed
        self.dg = self.problem.dg
        self.dh = self.problem.dh
        self.T = self.problem.T

        if self.use_momentum:
            self.first_moment = 0
            self.second_moment = 0

        # Get initial slack variable values
        N = x0.shape[0]
        if self.problem.dz > 0:
            z_init = self.problem.get_initial_z(x0)
            xuz = torch.cat((x0.clone(), z_init), dim=2).reshape(N, -1)
        else:
            xuz = x0.clone().reshape(N, -1)

        # driving force useful for helping exploration -- currently unused
        T = self.iters
        C = 1
        p = 1
        self.max_gamma = 1
        import time
        def driving_force(t):
            return ((t % (T / C)) / (T / C)) ** p

        if resample:
            xuz.data = self.resample(xuz.data)

        resample_period = 50
        path = [xuz.data.clone()]
        self.gamma = self.max_gamma
        for iter in range(T):
            # print(iter)
            # reset slack variables
            # if self.problem.dz > 0:
            #    xu = xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du]
            #    z_init = self.problem.get_initial_z(xu)
            #    xuz = torch.cat((xu, z_init), dim=2).reshape(N, -1)

            s = time.time()
            if T > 50:
                self.gamma = self.max_gamma * driving_force(iter)
            else:
                self.gamma = self.max_gamma

            # if (iter + 1) % resample_period == 0 and (iter < T - 1):
            #    xuz.data = self.resample(xuz.data)
            grad = self.compute_update(xuz)

            grad_norm = torch.linalg.norm(grad, keepdim=True, dim=-1)
            max_norm = 1
            grad = torch.where(grad_norm > max_norm, grad / grad_norm * max_norm, grad)

            xuz.data = xuz.data - self.dt * grad
            torch.nn.utils.clip_grad_norm_(xuz, 10)
            self._clamp_in_bounds(xuz)

            path.append(xuz.data.clone())

        # sort particles by penalty
        xuz = xuz.to(dtype=torch.float32)
        J = self.problem.get_cost(xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du])
        C, _, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1), compute_grads=False, compute_hess=False)
        penalty = J.reshape(N) + self.penalty * torch.sum(C.reshape(N, -1).abs(), dim=1)
        idx = torch.argsort(penalty, descending=False)
        path = torch.stack(path, dim=0).reshape(len(path), N, self.T, -1)[:, :, :, :self.dx + self.du]
        path = path[:, idx]
        return path
