import torch


class ValveDynamics(torch.nn.Module):
    """
    Assume the finger tip is having a sticking contact with the valve
    Assume the valve only turns in y axis
    Assume the finger tip location changes with the valve angle, rather than force. 
    """

    def __init__(self, dt, chain, inertia, valve_center):
        super().__init__()
        self.dt = dt
        self.chain = chain
        self.inertia = inertia
        self.valve_center = valve_center # the location of the root of the valve ? or the location of the center of valve

    def forward(self, state, control):
        """takes in finger tip position and action
        params: p: robot state but the finger states are finger ee loation [index ee location, thumb ee location, theta, theta dot]
        params: control: force applied at the finger tip
        """
        index_x, index_y, index_z, thumb_x, thumb_y, thumb_z, theta, w = torch.chunk(state, chunks=8, dim=-1)
        f_index_x, f_index_y, f_index_z, f_thumb_x, f_thumb_y, f_thumb_z = torch.chunk(control, chunks=6, dim=-1)
        index_ee_pos = torch.cat((index_x, index_y, index_z), dim=0)
        f_index = torch.cat((f_index_x, f_index_y, f_index_z), dim=0)
        f_thumb = torch.cat((f_thumb_x, f_thumb_y, f_thumb_z), dim=0)
        thumb_ee_pos = torch.cat((thumb_x, thumb_y, thumb_z), dim=0)
        total_torque = torch.cross(index_ee_pos - self.valve_center, f_index) + torch.cross(thumb_ee_pos - self.valve_center, f_thumb)
        total_torque = total_torque[1] # only consider the torque in y axis. 
        alpha = total_torque / self.inertia
        next_theta = theta + w * self.dt + 1/2 * alpha * self.dt**2
        next_w = w + alpha * self.dt
        # NOTE: Only returns the valve state for now, might need to return finger tip position as well. 
        return torch.cat((next_theta, next_w))
        
