
import numpy as np
import pickle as pkl

import torch
import time
import pathlib
from functools import partial

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd

import matplotlib.pyplot as plt# from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory, all_finger_constraints
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC
from scipy.spatial.transform import Slerp

from utils.allegro_utils import *

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# torch.cuda.set_device(1)
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroScrewdriver(AllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.screwdriver_force_balance:
            wrench_dim += 2
        # self.dg_per_t = self.num_fingers * (1 + 2)
        self.dg_per_t = self.num_fingers * (1 + 2 + 4) + wrench_dim
        self.dg_per_t += self.arm_dof
        self.dg_constant = -self.num_fingers # remove the kinematics constraint at the first step
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * self.num_fingers + 1 + self.num_fingers # one friction constraints per finger, num_finger means minimum force constraint
        if self.contact_region:
            self.dz += 1
        if self.collision_checking:
            self.dz += 2
        self.dh = self.dz * T  # inequality
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location, # the position of the root joint used to compute the torque balance, it's not necessarily the asset root pose. 
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 finger_stiffness=3,
                 arm_stiffness=None,
                 force_balance=False,
                 collision_checking=False,
                 device='cuda:0', 
                 obj_gravity=False,
                 contact_region=False,
                 **kwargs):
        self.num_fingers = len(fingers)
        self.obj_dof_code = [0, 0, 0, 1, 1, 1]
        self.obj_mass = 0.05
        self.contact_region = contact_region
        self.arm_dof = 0
        self.robot_dof = self.arm_dof + 4 * self.num_fingers
        du = self.robot_dof + 3 * self.num_fingers + 3 
        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, 
                                                 finger_stiffness=finger_stiffness, arm_stiffness=arm_stiffness, obj_dof_code=self.obj_dof_code, 
                                                 obj_joint_dim=1, screwdriver_force_balance=force_balance,
                                                 collision_checking=collision_checking, obj_gravity=obj_gravity,
                                                 contact_region=contact_region, du=du, device=device)
        self.min_force_dict = {'index': 0.0001, 'middle': 0.1, 'ring': 0.1, 'thumb': 0.1}
        # self.min_force_dict = {'index': 0.0001, 'middle': 1.0, 'ring': 1.0, 'thumb': 1.0}
        self.friction_coefficient = friction_coefficient
        self.friction_vel_constr = vmap(self._friction_vel_constr, randomness='same')
        self.grad_friction_vel_constr = vmap(jacrev(self._friction_vel_constr, argnums=(0, 1, 2)))
        if contact_region:
            self.index_contact_region_constr = vmap(self._index_contact_region_constr, randomness='same')
            self.grad_index_contact_region_constr = vmap(jacrev(self._index_contact_region_constr, argnums=(0, 1)))
        max_f = torch.ones(3) * 10
        min_f = torch.ones(3) * -10
        self.x_max = torch.cat((self.x_max, max_f))
        self.x_min = torch.cat((self.x_min, min_f))

        # for validity checking
        self.nominal_screwdriver_top = np.array([0, 0, 1.405]) # in the world frame
        self.env_force = True

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 1.5 * torch.randn(N, self.T, 3 * self.num_fingers + 3, device=self.device)
        force[:, :, :3] = force[:, :, :3] * 0.01 # NOTE: scale down the index finger force, might not apply to situations other than screwdriver
        force[:, :, -3:] = force[:, :, -3:] * 0.01 # expect the environment force to be small
        # force[:, :, 2] = -1 # index force pointing down
        # force[:, :, -1] = 1 # ground force pointing up
        force[:, :, 2] = -0.1 # index force pointing down
        force[:, :, -1] = 0.1 # ground force pointing up
        u = torch.cat((u, force), dim=-1)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :self.robot_dof] + u[:, t, :self.robot_dof]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (self.robot_dof + self.obj_dof):
            theta = []
            if self.obj_translational_dim > 0:
                current_obj_position = self.start[self.robot_dof: self.robot_dof + self.obj_translational_dim]
                theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
                theta.append(theta_position)
            if self.obj_rotational_dim > 0:
                current_obj_orientation = self.start[self.robot_dof + self.obj_translational_dim:self.robot_dof + self.obj_dof]
                current_obj_R = R.from_euler('XYZ', current_obj_orientation.cpu().numpy())
                goal_obj_R = R.from_euler('XYZ', self.goal[self.obj_translational_dim:self.obj_translational_dim + self.obj_rotational_dim].cpu().numpy())
                key_times = [0, self.T]
                times = np.linspace(0, self.T, self.T + 1)
                slerp = Slerp(key_times, R.concatenate([current_obj_R, goal_obj_R]))
                interp_rots = slerp(times)
                interp_rots = interp_rots.as_euler('XYZ')[1:]
                # current_obj_orientation = tf.euler_angles_to_matrix(current_obj_orientation, convention='XYZ')
                # current_obj_orientation = tf.matrix_to_rotation_6d(current_obj_orientation)
                theta.append(interp_rots)

            theta = np.concatenate(theta, axis=-1)
            theta = torch.tensor(theta, device=self.device, dtype=torch.float32)

            # repeat the current state
            # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((self.T, 1))
            theta = theta.unsqueeze(0).repeat((N,1,1))

            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + self.robot_dof]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        action_cost = 0
        # action_cost += 500 * torch.sum((action[1:, :self.arm_dof]) ** 2)
        # action_cost += torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 1 * torch.sum((state[1:, self.arm_dof:] - state[:-1, self.arm_dof:]) ** 2)
        smoothness_cost += 50 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)

        upright_cost = 10000 * torch.sum((state[:, -self.obj_dof:-1]) ** 2) # the screwdriver should only rotate in z direction

        goal_cost = torch.sum((500 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # add a running cost
        goal_cost += torch.sum((1 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))

        return smoothness_cost + action_cost + goal_cost + upright_cost
        
    def _index_repulsive(self, xu, link_name, compute_grads=True, compute_hess=False):
        """
        None teriminal link of the finger tip should have >= 0 sdf value
        """
        # print(xu[0, :2, 4 * self.num_fingers])
        N, T, _ = xu.shape
        # Retrieve pre-processed data
        ret_scene = self.data[link_name]
        g = -ret_scene.get('sdf').reshape(N, T + 1, 1) # - 0.0025
        grad_g_q = -ret_scene.get('grad_sdf', None)
        hess_g_q = ret_scene.get('hess_sdf', None)
        grad_g_theta = -ret_scene.get('grad_env_sdf', None)
        hess_g_theta = ret_scene.get('hess_env_sdf', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)
        if compute_grads:
            T_range = torch.arange(T, device=xu.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=xu.device)
            grad_g[:, T_range, T_range, :self.robot_dof] = grad_g_q[:, 1:]
            # is valve in state
            if self.dx == self.robot_dof + self.obj_dof:
                grad_g[:, T_range, T_range, self.robot_dof: self.robot_dof + self.obj_dof] = grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
    
    def _index_contact_region_constr(self, contact_pts, env_q):
        " contact pts are in the robot frame"
        " constraint specifying that the index finger only has contact with the top of the screwdriver"
        # N, T, _ = env_q.shape
        env_q = torch.cat((env_q, torch.zeros(1, device=env_q.device)), dim=-1) # add the screwdriver cap dim
        screwdriver_top_obj_frame = self.object_chain.forward_kinematics(env_q.unsqueeze(0))['screwdriver_cap']
        screwdriver_top_obj_frame = screwdriver_top_obj_frame.get_matrix().reshape(4, 4)[:3, 3]
        scene_trans = self.world_trans.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.object_asset_pos[0], self.object_asset_pos[1], self.object_asset_pos[2]))
        screwdriver_top_robot_frame = scene_trans.transform_points(screwdriver_top_obj_frame.reshape(-1, 3)).reshape(3)
        distance = torch.norm(contact_pts - screwdriver_top_robot_frame, dim=-1)
        h = distance - 0.02
        return h

    def _force_equlibrium_constraints_w_force(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        q = x[:, :-1, :self.robot_dof]
        next_q = x[:, 1:, :self.robot_dof]
        next_env_q = x[:, 1:, self.robot_dof:self.robot_dof + self.obj_dof]
        u = xu[:, :, self.dx: self.dx + self.robot_dof]
        force = xu[:, :, self.dx + self.robot_dof: self.dx + self.robot_dof + 3 * self.num_fingers + 3]
        # the contact point does not include the env contact point as this is the special case. We choose the reference point for torque
        # right at the env contact points, thus we only need to reason about the force from env contacts
        force_list = force.reshape((force.shape[0], force.shape[1], self.num_fingers + 1, 3))
        # contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1].reshape(-1, 3, self.robot_dof)\
        #                      for finger_name in self.fingers]
        contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:].reshape(-1, 3, self.robot_dof)\
                             for finger_name in self.fingers]
        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=xu.device)
        contact_point_list = [self.data[finger_name]['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3) for finger_name in self.fingers]
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=xu.device)

        g = self.force_equlibrium_constr(q.reshape(-1, self.robot_dof), 
                                         u.reshape(-1, self.robot_dof), 
                                         next_q.reshape(-1, self.robot_dof), 
                                         force_list.reshape(-1, self.num_fingers + 1, 3),
                                         contact_jac_list,
                                         contact_point_list,
                                         next_env_q.reshape(-1, self.obj_dof)).reshape(N, T, -1)
        # print(g.abs().max().detach().cpu().item(), g.abs().mean().detach().cpu().item())
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact, dg_dnext_env_q = self.grad_force_equlibrium_constr(q.reshape(-1, self.robot_dof), 
                                                                                  u.reshape(-1, self.robot_dof), 
                                                                                  next_q.reshape(-1, self.robot_dof), 
                                                                                  force_list.reshape(-1, self.num_fingers + 1, 3),
                                                                                  contact_jac_list,
                                                                                  contact_point_list,
                                                                                  next_env_q.reshape(-1, self.obj_dof))
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], self.num_fingers * 3 + 3)
            
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            dg_dq = dg_dq.reshape(N, T, g.shape[2], self.robot_dof) 
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], self.robot_dof) 
            for i, finger_name in enumerate(self.fingers):
                # djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, self.robot_dof, self.robot_dof)[:, :-1] # jacobian is the contact jacobian
                # dg_dq = dg_dq + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, self.robot_dof)
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, self.robot_dof, self.robot_dof)[:, 1:]
                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dnext_q.reshape(N, T, -1, self.robot_dof)
                
                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1]
                dg_dq = dg_dq + dg_dcontact[:, : ,i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq 
            grad_g[:, :, T_plus, T_minus, :self.robot_dof] = dg_dq.reshape(N, T, g.shape[2], self.robot_dof)[:, 1:].transpose(1, 2)  # first q is the start
            dg_du = torch.cat((dg_du, dg_dforce), dim=-1)  # check the dim
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_range, T_range, :self.robot_dof] = dg_dnext_q.reshape(N, T, -1, self.robot_dof).transpose(1, 2)
            if self.obj_gravity:
                grad_g[:, :, T_range, T_range, self.robot_dof: self.robot_dof + self.obj_dof] = dg_dnext_env_q.reshape(N, T, -1, self.obj_dof).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None
        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None
        
    def _index_contact_region_constraint(self, xu, compute_grads=True, compute_hess=False):
        " constraint specifying that the index finger only has contact with the top of the screwdriver"
        N, T, d = xu.shape
        env_q = xu[:, :, self.robot_dof: self.robot_dof + self.obj_dof].reshape(-1, self.obj_dof)
        # Retrieve pre-processed data
        ret_scene = self.data['index']
        index_contact_pts = ret_scene['closest_pt_world'].reshape(N, T + 1, 3)[:, 1:]
        h = self.index_contact_region_constr(index_contact_pts.reshape(-1, 3), env_q.reshape(-1, self.obj_dof)).reshape(N, -1)
        if compute_grads:
            grad_h = torch.zeros(N, 1, T, T, d, device=self.device)
            dh_dcontact, dh_denv_q = self.grad_index_contact_region_constr(index_contact_pts.reshape(-1, 3), env_q.reshape(-1, self.obj_dof))
            dcontact_dq = ret_scene['closest_pt_q_grad'].reshape(N, T+1, 3, self.robot_dof)[:, 1:]
            dh_dq = dh_dcontact.reshape(N, T, 1, 3) @ dcontact_dq.reshape(N, T, 3, self.robot_dof)

            T_range = torch.arange(T, device=xu.device)
            grad_h[:, :, T_range, T_range, :self.robot_dof] = dh_dq.reshape(N, T, 1, self.robot_dof).transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.robot_dof: self.robot_dof + self.obj_dof] = dh_denv_q.reshape(N, T, 1, self.obj_dof).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

    
    def _friction_vel_constr(self, dq, contact_normal, contact_jacobian):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector

        # compute the force in robot frame
        # force = (torch.linalg.lstsq(contact_jacobian.transpose(-1, -2),
        #                            dq.unsqueeze(-1))).solution.squeeze(-1)
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        # transform contact normal to world frame
        contact_normal_world = self.world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # transform force to contact frame
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0)
        # force_contact_frame = R.transpose(0, 1) @ force_world_frame.unsqueeze(-1)
        B = self.get_friction_polytope().detach()

        # compute contact point velocity in contact frame
        contact_v_contact_frame = R.transpose(0, 1) @ self.world_trans.transform_normals(
                (contact_jacobian @ dq).unsqueeze(0)).squeeze(0)
        # TODO: there are two different ways of doing a friction cone
        # Linearized friction cone - but based on the contact point velocity
        # force is defined as the force of robot pushing the object
        return B @ contact_v_contact_frame

    @all_finger_constraints
    def _friction_vel_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        u = xu[:, :, self.dx:]
        u = u[:, :, :self.robot_dof].reshape(-1, self.robot_dof)

        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1] # contact normal is pointing out 
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        # compute constraint value
        h = self.friction_vel_constr(u,
                                 contact_normal.reshape(-1, 3),
                                 contact_jac.reshape(-1, 3, self.robot_dof)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal, dh_djac = self.grad_friction_vel_constr(u,
                                                                   contact_normal.reshape(-1, 3),
                                                                   contact_jac.reshape(-1, 3, self.robot_dof))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, self.robot_dof, self.robot_dof)[:, :-1]

            dh = dh_dnormal.shape[1]
            dh_dq = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, self.robot_dof)
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :self.robot_dof] = dh_dq[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range_plus, T_range_minus, self.robot_dof: self.robot_dof + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx: self.dx + self.robot_dof] = dh_du.reshape(N, T, dh, self.robot_dof).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None
    def _env_force_constraint(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]
        u = xu[:, :, self.dx:]
        env_force = xu[:, :, -3:]
        env_force_z = env_force[:, :, 2]
        h = -env_force_z
        if compute_grads:
            grad_h = torch.zeros(N, 1, T, T, d, device=self.device)
            grad_h[:, :, torch.arange(T), torch.arange(T), -3:] = -1
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None
    
    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        h_force, grad_h_force, hess_h_force = self._min_force_constraints(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        # h_vel, grad_h_vel, hess_h_vel = self._friction_vel_constraint(
        #     xu=xu.reshape(-1, T, self.dx + self.du),
        #     compute_grads=compute_grads,
        #     compute_hess=compute_hess)
        if self.collision_checking:
            h_rep_1, grad_h_rep_1, hess_h_rep_1 = self._index_repulsive(
                xu=xu.reshape(-1, T, self.dx + self.du),
                link_name='allegro_hand_hitosashi_finger_finger_link_2',
                compute_grads=compute_grads,
                compute_hess=compute_hess)
            
            h_rep_2, grad_h_rep_2, hess_h_rep_2 = self._index_repulsive(
                xu=xu.reshape(-1, T, self.dx + self.du),
                link_name='allegro_hand_hitosashi_finger_finger_link_3',
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        
            # h = torch.cat((h, h_vel), dim=1)
            h_rep = torch.cat((h_rep_1, h_rep_2), dim=1)
        if self.contact_region:
            h_con_region, grad_h_con_region, hess_h_con_region = self._index_contact_region_constraint(
                xu=xu.reshape(-1, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        h_env, grad_h_env, hess_h_env = self._env_force_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            print(f"max min force constraint: {torch.max(h_force)}")
            if self.collision_checking:
                print(f"max index repulsive constraint: {torch.max(h_rep)}")
            if self.contact_region:
                print(f"max contact region constraint: {torch.max(h_con_region)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            print(f"max env force constraint: {torch.max(h_env)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
            result_dict['env_force'] = torch.max(h_env).item()
            result_dict['env_force_mean'] = torch.mean(h_env).item()
            if self.collision_checking:
                result_dict['index_rep'] = torch.max(h_rep).item()
                result_dict['index_rep_mean'] = torch.mean(h_rep).item()
            if self.contact_region:
                result_dict['contact_region'] = torch.max(h_con_region).item()
                result_dict['contact_region_mean'] = torch.mean(h_con_region).item()
            # result_dict['singularity'] = torch.max(h_sin).item()
            return result_dict

        # h = torch.cat((h,
        #             #    h_step_size,
        #                h_sin), dim=1)
        h = torch.cat((h, h_force), dim=1)
        h = torch.cat((h, h_env), dim=1)
        if self.collision_checking:
            h = torch.cat((h, h_rep), dim=1)  
        if self.contact_region:
            h = torch.cat((h, h_con_region), dim=1)
        if compute_grads:
            # NOTE the order of the gradients for different constraints should match the order defined in the value
            grad_h = torch.cat((grad_h, grad_h_force), dim=1)
            grad_h = torch.cat((grad_h, grad_h_env), dim=1)
            # grad_h = torch.cat((grad_h, 
            #                     # grad_h_step_size,
            #                     grad_h_sin), dim=1)
            if self.collision_checking:
                grad_h_rep = torch.cat((grad_h_rep_1, grad_h_rep_2), dim=1)
                grad_h = torch.cat((grad_h, grad_h_rep), dim=1)
            if self.contact_region:
                grad_h = torch.cat((grad_h, grad_h_con_region), dim=1)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None
    
    def check_validity(self, state):
        screwdriver_state = state[-self.obj_dof:]
        screwdriver_top_pos = get_screwdriver_top_in_world(screwdriver_state, self.object_chain, self.world_trans, self.object_asset_pos)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - self.nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        else:
            validity_flag = True
        return validity_flag
    
    