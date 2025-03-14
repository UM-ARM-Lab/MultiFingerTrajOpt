import torch
import numpy as np
from ccai.gp import GPSurfaceModel, get_random_surface, get_random_obs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!


class QuadrotorEnv:

    def __init__(self, randomize_GP=False, surface_data_fname=None, obstacle_data_fname=None, obstacle_mode=None,
                 surface_constraint=True):
        assert obstacle_mode in [None, 'static', 'dynamic', 'gp']
        self.env_dims = [-10, 10]
        self.state_dim = 12
        self.dt = 0.1
        self.state = None
        self.surface_model = None
        if surface_constraint:
            if not randomize_GP and surface_data_fname is not None:
                data = np.load(surface_data_fname)
                self.surface_model = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32),
                                                    torch.from_numpy(data['z']).to(dtype=torch.float32))
            else:
                self.surface_model = get_random_surface()

        self.surface_plotting_vars = self._get_plotting_vars(self.surface_model)

        self.obstacle_pos = np.array([0.0, 0.0])
        self.obstacle_r = 1
        self.obstacle_mode = obstacle_mode
        self.reset()

        if self.obstacle_mode == 'gp':
            if obstacle_data_fname is None:
                self.obstacle_model = get_random_obs(self.state, self.goal)
            else:
                data = np.load(obstacle_data_fname)
                self.obstacle_model = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32),
                                                     torch.from_numpy(data['z']).to(dtype=torch.float32))

            self.obs_plotting_vars = self._get_plotting_vars(self.obstacle_model)
        else:
            self.obstacle_model = None

        self.alpha = 0.5
        self.ax = None
        self._surf = None
        self._render_pos = None

    def _get_plotting_vars(self, gp_model=None):
        # For rendering height
        N = 100
        xs = torch.linspace(-6, 6, steps=N)
        ys = torch.linspace(-6, 6, steps=N)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        test_x = torch.stack((x.flatten(), y.flatten()), dim=1)
        if gp_model is not None:
            z, _, _, = gp_model.posterior_mean(test_x)
            z = z.detach()
        else:
            z = torch.zeros_like(x)
        x = test_x[:, 0].reshape(N, N).numpy()
        y = test_x[:, 1].reshape(N, N).numpy()
        z = z.reshape(N, N).numpy()
        return x, y, z

    def _get_surface_h(self, state):
        xy = torch.from_numpy(state[:2]).reshape(1, 2).to(dtype=torch.float32)
        with torch.no_grad():
            z, _, _ = self.surface_model.posterior_mean(xy)
        return z.numpy().reshape(1).astype(np.float32)

    def _get_gp_obs_sdf(self, state):
        xy = torch.from_numpy(state[:2]).reshape(1, 2).to(dtype=torch.float32)
        with torch.no_grad():
            z, _, _ = self.obstacle_model.posterior_mean(xy)
        return z.numpy().reshape(1).astype(np.float32)

    def reset(self):
        start = np.zeros(self.state_dim)
        goal = np.zeros(3)

        # positions
        got_sg = False
        while not got_sg:
            start[:2] = 10 * np.random.rand(2) - 5  #
            start[:2] = np.array([-3.0, -3.0]) - 1.5 * np.random.rand(2)
            goal[:2] = np.array([4, 4])
            #goal = 10 * np.random.rand(3) - 5

            if np.linalg.norm(start[:2] - goal[:2]) > 4:
                got_sg = True

        if self.surface_model is not None:
            start[2] = self._get_surface_h(start)
            goal[2] = self._get_surface_h(goal)
        else:
            start[2] = np.random.rand(1)
            goal[2] = np.random.rand(1)

        """
        # positions
        start[:2] = np.array([-4, -4]) - 1. * np.random.rand(2)
        start[2] = self._get_surface_h(start)
        self.goal = np.array([4, 4, 0])
        self.goal = self.goal + 0.5 * np.random.randn(3)
        """
        # Angles in rad
        start[3:6] = 0.05 * (-np.pi + 2 * np.pi * np.random.rand(3))

        # Initial velocity
        start[6:] = 0.05 * np.random.randn(6)

        # start[9:] *= 5
        self.state = start
        self.goal = goal
        if self.obstacle_mode == 'dynamic':
            self.obstacle_pos = np.array([-1.0, 1.5])
        else:
            self.obstacle_pos = np.array([0.0, 0.0])

    def get_constraint_violation(self):
        constraints = {}

        if self.surface_model is not None:
            surface_h = self._get_surface_h(self.state)
            surface_violation = self.state[2] - surface_h
            constraints['surface'] = surface_violation.item()

        if self.obstacle_mode is not None:
            if self.obstacle_mode == 'gp':
                sdf = self._get_gp_obs_sdf(self.state)
                #print(sdf)
                #print(self.state[:2])
                obstacle_violation = np.clip(sdf, a_min=0, a_max=None)
            else:
                obstacle_violation = self.obstacle_r ** 2 - np.sum((self.state[:2] - self.obstacle_pos) ** 2)
                obstacle_violation = np.clip(obstacle_violation, a_min=0, a_max=None)
            constraints['obstacle'] = obstacle_violation.item()
        #print(constraints)
        return constraints

    def step(self, control):

        # Unroll state
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 1
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = self.state

        u1, u2, u3, u4 = control

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = np.cos(phi)
        ctheta = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        stheta = np.sin(theta)
        spsi = np.sin(psi)
        ttheta = np.tan(theta)

        ''' accelerations first '''
        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz

        # symplectic euler -- update velocities first
        new_xdot = x_dot + x_ddot * self.dt
        new_ydot = y_dot + y_ddot * self.dt
        new_zdot = z_dot + z_ddot * self.dt
        new_p = p + p_dot * self.dt
        new_q = q + q_dot * self.dt
        new_r = r + r_dot * self.dt

        ''' velocities'''
        psi_dot = new_q * sphi / ctheta + new_r * cphi / ctheta
        theta_dot = new_q * cphi - new_r * sphi
        phi_dot = new_p + new_q * sphi * ttheta + new_r * cphi * ttheta

        new_phi = phi + phi_dot * self.dt
        new_theta = theta + theta_dot * self.dt
        new_psi = psi + psi_dot * self.dt
        new_x = x + new_xdot * self.dt
        new_y = y + new_ydot + self.dt
        new_z = z + new_zdot + self.dt

        # dstate = np.stack((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
        #                   x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), axis=-1)

        # self.state = self.state + dstate * self.dt
        self.state = np.stack(
            (new_x, new_y, new_z, new_phi, new_theta, new_psi, new_xdot, new_ydot, new_zdot, new_p, new_q, new_r)
        )
        # self.state[3:6] = normalize_angles(self.state[3:6])

        if self.obstacle_mode == 'dynamic':
            # if self.obstacle_pos[0] < 3:
            self.obstacle_pos += np.array([0.3, -0.15])

        return self.state, self.get_constraint_violation()

    def _get_surface_colours(self):
        x, y, z = self.surface_plotting_vars
        N = x.shape[0]
        if self.obstacle_mode is None:
            c = np.array([0.0, 0.0, .7, self.alpha])[None, None, :]
            return c.repeat(N, axis=0).repeat(N, axis=1)

        if self.obstacle_mode == 'gp':
            _, _, zobs = self.obs_plotting_vars
            in_obs = np.where(zobs > 0, 1, 0)
        else:
            in_obs = np.where((x - self.obstacle_pos[0]) ** 2 + (y - self.obstacle_pos[1]) ** 2 < self.obstacle_r ** 2,
                              1, 0)

        c = np.where(in_obs[:, :, None],
                     np.array([1.0, 0.0, 0.0, self.alpha])[None, None, :],
                     np.array([0, 0.0, .7, self.alpha])[None, None, :]
                     )
        return c

    def render_init(self):
        self.ax = plt.axes(projection='3d')

        self._surf = self.ax.plot_surface(self.surface_plotting_vars[0],
                                          self.surface_plotting_vars[1],
                                          self.surface_plotting_vars[2] - 0.1, alpha=self.alpha,
                                          shade=True,
                                          antialiased=True, linewidth=0,
                                          facecolors=self._get_surface_colours(),
                                          rstride=1,
                                          cstride=1)
        self._render_pos = self.ax.scatter(self.state[0], self.state[1], self.state[2], s=100, c='g')

        self.ax.scatter(self.goal[0], self.goal[1], self.goal[2], s=100, c='k')
        self.ax.view_init(60, -50)
        # self.ax.view_init(90, -50)
        self.ax.axes.set_xlim3d(left=-6, right=6)
        self.ax.axes.set_ylim3d(bottom=-6, top=6)
        self.ax.axes.set_zlim3d(bottom=-3, top=3)
        self.ax.set_xlabel('$x$', fontsize=20)
        self.ax.set_ylabel('$y$', fontsize=20)
        self.ax.set_zlabel('$z$', fontsize=20)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        return self.ax

    def render_update(self):
        #print(self.state[:2], 'rendering')
        self._render_pos._offsets3d = ([self.state[0]], [self.state[1]], [self.state[2]])
        if self.obstacle_mode == 'dynamic':
            self._surf.set_facecolors(self._get_surface_colours()[:-1, :-1].reshape(-1, 4))

    def get_constraint_params(self):
        params = {
            'surface': None,
            'obstacle': None
        }
        if self.surface_model is not None:
            params['surface'] = self.surface_model.train_y.reshape(-1).cpu().numpy()

        if self.obstacle_model is not None:
            params['obstacle'] = self.obstacle_model.train_y.reshape(-1).cpu().numpy()

        return params


if __name__ == "__main__":
    env = QuadrotorEnv(randomize_GP=False, surface_data_fname='../examples/surface_data.npz', obstacle_mode='gp')
    for i in range(100):
        env.state = np.zeros(12)
        env.state[:2] = np.array([-4, -4])
        env.goal[:2] = np.array([4, 4])
        env.obstacle_model = get_random_obs(env.state, env.goal)
        env.obs_plotting_vars = env._get_plotting_vars(env.obstacle_model)
        ax = env.render_init()

        plt.savefig(f'obstacle_data_{i}.png')

        data = {
            'xy': env.obstacle_model.train_x.numpy(),
            'z': env.obstacle_model.train_y.numpy()
        }
        np.savez(f'obstacle_data_{i}.npz', **data)