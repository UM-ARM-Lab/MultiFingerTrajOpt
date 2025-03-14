import torch
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt


class Constrained_SVGD_MPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.N = params.get('N')
        self.resample_steps = params.get('resample_steps', 1)
        self.problem = problem
        self.solver = ConstrainedSteinTrajOpt(problem, params)
        self.random_init_every_iter = params.get('random_init_every_iter', False)

        # initialize randomly
        self.x = self.problem.get_initial_xu(self.N)
        self.warmed_up = False
        self.iter = 0

    def step(self, state, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T

        self.problem.update(state, T=new_T, **kwargs)
        # warm starting
        if self.warmed_up:
            self.solver.iters = self.online_iters
            resample = True if (self.iter + 1) % self.resample_steps == 0 else False
        else:
            self.solver.iters = self.warmup_iters
            self.warmed_up = True
            resample = False
        if self.random_init_every_iter:
            self.x = self.problem.get_initial_xu(self.N)
        path = self.solver.solve(self.x, resample)
        self.x = path[-1]
        self.iter += 1
        best_trajectory = self.x[0].clone()
        all_trajectories = self.x.clone()
        self.shift()
        return best_trajectory, all_trajectories

    def shift(self):
        if self.fix_T:
            self.x = torch.roll(self.x, shifts=-1, dims=1)
            self.x[:, -1] = self.x[:, -2]  # just copy over previous last
        else:
            self.x = self.x[:, 1:]
        #self.x = self.problem.shift(self.x)

    def reset(self, start, initial_x=None, **kwargs):
        self.problem.update(start, **kwargs)
        self.warmed_up = False
        self.iter = 0
        self.x = self.problem.get_initial_xu(self.N)
        if initial_x is not None:
            self.x = initial_x
            self.warmup_iters = 10
