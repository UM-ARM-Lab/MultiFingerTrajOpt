import torch


class Quadrotor12DDynamics(torch.nn.Module):

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def forward(self, state, control):
        ''' unroll state '''
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 1
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = torch.chunk(state, chunks=12, dim=-1)

        u1, u2, u3, u4 = torch.chunk(control, chunks=4, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = torch.cos(phi)
        ctheta = torch.cos(theta)
        cpsi = torch.cos(psi)
        sphi = torch.sin(phi)
        stheta = torch.sin(theta)
        spsi = torch.sin(psi)
        ttheta = torch.tan(theta)

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

        return torch.cat(
            (new_x, new_y, new_z, new_phi, new_theta, new_psi, new_xdot, new_ydot, new_zdot, new_p, new_q, new_r),
            dim=-1
        )
        # self.state[3:6] = normalize_angles(self.state[3:6])
        # dstate = torch.cat((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
        #                    x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), dim=-1)
        #
        # return state + dstate * self.dt
