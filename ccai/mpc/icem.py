import torch
import colorednoise


class iCEM:

    def __init__(self, problem, params):
        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.H = self.problem.T
        self.fixed_H = params.get('receding_self.H', True)
        self.N = params.get('N', 100)
        self.device = params.get('device', 'cuda:0')
        self.sigma = params.get('sigma', [1.0] * self.du)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.online_iters = params.get('online_iters', 100)
        self.includes_x0 = params.get('include_x0', False)
        self.noise_beta = params.get('noise_beta', 3)
        self.K = params.get('num_elites', 10)
        self.alpha = params.get('alpha', 0.05)
        self.keep_fraction = params.get('elites_keep_fraction', 0.5)

        self.sigma = torch.tensor(self.sigma).to(device=self.device)
        self.std = self.sigma.clone()
        # initialise mean and std of actions
        self.mean = torch.zeros(self.H, self.du, device=self.device)
        self.kept_elites = None
        self.warmed_up = False

    def reset(self):
        self.warmed_up = False
        self.mean = torch.zeros(self.H, self.du, device=self.device)
        self.std = self.sigma.clone()
        self.kept_elites = None

    def sample_action_sequences(self, state, N):
        # colored noise
        if self.noise_beta > 0:
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(N, self.du,
                                                                                self.H)).transpose(
                [0, 2, 1])
            samples = torch.from_numpy(samples).to(device=self.device).float()
        else:
            samples = torch.randn(N, self.H, self.du, device=self.device).float()

        U = self.mean + self.std * samples
        return U

    def update_distribution(self, elites):
        """
        param: elites - K x H x du number of best K control sequences by cost
        """

        # fit around mean of elites
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0)

        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std

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

        # Shift the keep elites

        for i in range(iterations):
            if self.kept_elites is None:
                # Sample actions
                U = self.sample_action_sequences(x, self.N)
            else:
                # reuse the elites from the previous iteration
                U = self.sample_action_sequences(x, self.N - len(self.kept_elites))
                U = torch.cat((U, self.kept_elites), dim=0)

            # evaluate costs and update the distribution!
            pred_x = self._rollout_dynamics(x, U)
            costs = self._cost(pred_x, U)
            sorted, indices = torch.sort(costs)
            elites = U[indices[:self.K]]
            self.update_distribution(elites)
            # save kept elites fraction
            self.kept_elites = U[indices[:int(self.K * self.keep_fraction)]]

        # Return best sampled trajectory
        out_U = elites[0].clone()
        out_X = self._rollout_dynamics(x, out_U.reshape(1, self.H, self.du)).reshape(self.H, self.dx)
        out_trajectory = torch.cat((out_X, out_U), dim=-1)

        # Top N // 20 sampled trajectories - for visualization
        sampled_trajectories = torch.cat((pred_x, U), dim=-1)
        # only return best 10% trajectories for visualization
        sampled_trajectories = sampled_trajectories[torch.argsort(costs, descending=False)][:64]

        self.shift()
        return out_trajectory, sampled_trajectories

    def shift(self):
        # roll distribution
        self.mean = torch.roll(self.mean, -1, dims=0)
        self.mean[-1] = torch.zeros(self.du, device=self.device)
        self.std = self.sigma.clone()
        # Also shift the elites
        if self.kept_elites is not None:
            self.kept_elites = torch.roll(self.kept_elites, -1, dims=1)
            self.kept_elites[:, -1] = self.sigma * torch.randn(len(self.kept_elites), self.du, device=self.device)
