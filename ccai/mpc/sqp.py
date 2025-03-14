import torch
# import nlopt
import numpy as np


class SQPMPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.problem = problem

        # initialize randomly
        self.x = self.problem.get_initial_xu(1).squeeze(0)
        # store if problem has equality and inequality constraints
        self.ineq = False if self.problem.dh == 0 else True
        self.eq = False if self.problem.dh == 0 else True

        self.warmed_up = False

    def step(self, state, **kwargs):
        if self.fix_T or (not self.warmed_up):
            new_T = None
        else:
            new_T = self.problem.T - 1

        self.problem.update(state, T=new_T, **kwargs)

        if self.warmed_up:
            iters = self.online_iters
        else:
            iters = self.warmup_iters
        self.warmed_up = True

        # set up the optimization problem
        opt = nlopt.opt(nlopt.LD_SLSQP, (self.problem.dx + self.problem.du) * self.problem.T)
        opt.set_min_objective(self.problem.objective)

        lower, upper = self.problem.get_bounds()
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)

        opt.add_inequality_mconstraint(self.problem.con_ineq, np.zeros(self.problem.dh))
        opt.add_equality_mconstraint(self.problem.con_eq, np.zeros(self.problem.dg))
        opt.set_maxeval(iters)
        # opt.set_ftol_abs(1e-4)
        opt.set_xtol_abs(1e-6)
        x = self.x.numpy().reshape(-1)
        # initial x must be within bounds
        x = np.clip(x, lower, upper)
        # perform the optimization
        try:
            xopt = opt.optimize(x)
        except Exception as e:
            print(e)
            print('retrying with perturbed initialization')
            x = x + np.random.randn(x.shape[0]) * 1e-2
            x = np.clip(x, lower, upper)
            xopt = opt.optimize(x)
        self.x = torch.from_numpy(xopt).reshape(self.problem.T, -1).to(dtype=torch.float32)
        ret_x = self.x.clone()
        self.shift()

        return ret_x, ret_x.unsqueeze(0)

    def shift(self):
        # if self.fix_T:
        #    self.x = torch.roll(self.x, shifts=-1, dims=0)
        #    self.x[-1] = self.x[-2]
        # else:
        #    self.x = self.x[1:]
        self.x = self.problem.shift(self.x.unsqueeze(0)).squeeze(0)

    def reset(self, start, **kwargs):
        self.problem.update(start, **kwargs)
        self.warmed_up = False
