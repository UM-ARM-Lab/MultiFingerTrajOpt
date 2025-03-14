import torch
import cyipopt


class IpoptMPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.problem = problem
        # self.solver = ConstrainedSteinTrajOpt(problem, params)

        # initialize randomly
        self.x = self.problem.get_initial_xu(1).squeeze(0)
        g = self.problem.con_eq(self.x.numpy())
        h = self.problem.con_ineq(self.x.numpy())
        # store if problem has equality and inequality constraints
        self.ineq = False if h is None else True
        self.eq = False if g is None else True

        self.warmed_up = False

    def step(self, state, **kwargs):
        if self.fix_T or (not self.warmed_up):
            new_T = None
        else:
            new_T = self.problem.T - 1

        self.problem.update(state, T=new_T, **kwargs)

        # constraints
        cons = []
        if self.eq is True:
            cons.append(
                {'type': 'eq', 'fun': self.problem.con_eq,
                 'jac': self.problem.con_eq_grad,
                 'hess': self.problem.con_eq_hvp}
            )
        if self.ineq is True:
            cons.append(
                {'type': 'ineq', 'fun': self.problem.con_ineq,
                 'jac': self.problem.con_ineq_grad,
                 'hess': self.problem.con_ineq_hvp}
            )

        lower, upper = self.problem.get_bounds()
        bnds = [(u, l) for u, l in zip(lower, upper)]

        if self.warmed_up:
            iters = self.online_iters
        else:
            iters = self.warmup_iters
        self.warmed_up = True

        x = self.x.numpy().reshape(-1)
        res = cyipopt.minimize_ipopt(self.problem.objective, jac=self.problem.objective_grad,
                                     hess=self.problem.objective_hess, x0=x,
                                     bounds=bnds,
                                     constraints=cons, options={'disp': 0,
                                                                'max_iter': iters,
                                                                'tol': 1e-4,
                                                                'acceptable_tol': 1e-4,
                                                                'hessian_approximation': 'limited-memory'})

        self.x = torch.from_numpy(res.x).reshape(self.problem.T, -1).to(dtype=torch.float32)
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
