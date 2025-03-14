import torch


class MPPI:
    def __init__(self, problem, params):
        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.H = self.problem.T
        self.fixed_H = params.get('receding_horizon', True)
        self.N = params.get('N', 100)
        self.device = params.get('device', 'cuda:0')
        self.sigma = params.get('sigma', [1.0] * self.du)
        self.lambda_ = params.get('lambda', 0.1)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.online_iters = params.get('online_iters', 100)
        self.includes_x0 = params.get('include_x0', False)

        self.sigma = torch.tensor(self.sigma, device=self.device)

        # randomly generate actions
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)

        self.warmed_up = False

    def _cost(self, x, u):
        xu = torch.cat((x, u), dim=-1)
        return self.problem.objective(xu)

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

    def step(self, x, **kwargs):
        if self.fixed_H or (not self.warmed_up):
            new_T = None
        else:
            new_T = self.problem.T - 1
            self.H = new_T

        self.problem.update(x, T=new_T, **kwargs)

        if self.warmed_up:
            iterations = self.online_iters
        else:
            iterations = self.warmup_iters
            self.warmed_up = True

        for iter in range(iterations):
            # Sample peturbations
            noise = torch.randn(self.N, self.H, self.du, device=self.device)
            peturbed_actions = self.U.unsqueeze(dim=0) + self.sigma * noise
            action_cost = torch.sum(self.lambda_ * noise * self.U / self.sigma ** 2, dim=[1, 2])

            pred_x = self._rollout_dynamics(x, peturbed_actions)
            # Get total cost
            total_cost = self._cost(pred_x, peturbed_actions)
            total_cost += action_cost
            total_cost -= torch.min(total_cost)
            total_cost /= torch.max(total_cost)
            omega = torch.softmax(-total_cost / self.lambda_, dim=0)
            self.U = torch.sum((omega.reshape(-1, 1, 1) * peturbed_actions), dim=0)

        out_U = self.U.clone()
        out_X = self._rollout_dynamics(x, out_U.reshape(1, self.H, self.du)).reshape(self.H, self.dx)
        out_trajectory = torch.cat((out_X, out_U), dim=-1)
        sampled_trajectories = torch.cat((pred_x, peturbed_actions), dim=-1)
        # only return best 10% trajectories for visualization
        sampled_trajectories = sampled_trajectories[torch.argsort(total_cost, descending=False)][:32]

        self.shift()


        return out_trajectory, sampled_trajectories

    def shift(self):
        if self.fixed_H:
            self.U = torch.roll(self.U, shifts=-1, dims=0)
            self.U[-1] = self.sigma * torch.randn(self.du, device=self.device)
        else:
            self.U = self.U[1:]

    def reset(self):
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)
        self.warmed_up = False
