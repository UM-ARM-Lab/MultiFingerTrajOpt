from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import copy
import yaml
import pathlib
from functools import partial

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd

import matplotlib.pyplot as plt
from utils.allegro_utils import partial_to_full_state, full_to_partial_state, state2ee_pos, visualize_trajectory, all_finger_constraints
# from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory, all_finger_constraints
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# torch.cuda.set_device(1)
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

def vector_cos(a, b):
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))
class AllegroScrewdriver(AllegroValveTurning):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 force_balance=False,
                 collision_checking=False,
                 obj_gravity=False,
                 contact_region=False,
                 static_init=False, # whether to initialize the screwdriver as the current state
                 use_arm=False,
                 arm_stiffness=None,
                 finger_stiffness=None,
                 device='cuda:0', **kwargs):
        self.num_fingers = len(fingers)
        self.obj_dof_code = [0, 0, 0, 1, 1, 1]
        self.optimize_force = optimize_force
        self.obj_mass = 0.1
        self.contact_region = contact_region
        self.static_init = static_init
        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=self.obj_dof_code, 
                                                 obj_joint_dim=1, optimize_force=optimize_force, 
                                                 screwdriver_force_balance=force_balance,
                                                 collision_checking=collision_checking, obj_gravity=obj_gravity,
                                                 contact_region=contact_region, use_arm=use_arm, arm_stiffness=arm_stiffness, 
                                                 finger_stiffness=finger_stiffness, device=device)
        self.friction_coefficient = friction_coefficient
        self.default_index_ee_loc_in_screwdriver = torch.tensor([0.0087, -0.016, 0.1293], device=device).unsqueeze(0)
        self.friction_vel_constr = vmap(self._friction_vel_constr, randomness='same')
        self.grad_friction_vel_constr = vmap(jacrev(self._friction_vel_constr, argnums=(0, 1, 2)))
        if contact_region:
            self.index_contact_region_constr = vmap(self._index_contact_region_constr, randomness='same')
            self.grad_index_contact_region_constr = vmap(jacrev(self._index_contact_region_constr, argnums=(0, 1)))

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        if self.optimize_force:
            u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
            if self.use_arm:
                arm_u = 0.002 * torch.randn(N, self.T, 7, device=self.device)
                u = torch.cat((arm_u, u), dim=-1)
            force = 1.5 * torch.randn(N, self.T, 3 * self.num_fingers, device=self.device)
            force[:, :, :3] = force[:, :, :3] * 0.01 # NOTE: scale down the index finger force, might not apply to situations other than screwdriver
            u = torch.cat((u, force), dim=-1)
        else:
            u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :self.robot_dof] + u[:, t, :self.robot_dof]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (self.robot_dof + self.obj_dof):
            if self.obj_dof == 6:
                current_obj_position = self.start[self.robot_dof: self.robot_dof + self.obj_translational_dim]
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

                theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
                theta = np.concatenate((theta_position, interp_rots), axis=-1)
                theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            else:
                if not self.static_init:
                    theta = np.linspace(self.start[-self.obj_dof:].cpu().numpy(), self.goal.cpu().numpy(), self.T + 1)[1:]
                    theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
                else:
                    # repeat the current state
                    theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((self.T, 1))
            theta = theta.unsqueeze(0).repeat((N,1,1))

            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + self.robot_dof]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        if self.optimize_force:
            action_cost = 0
        else:
            action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)
        # action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        smoothness_cost += 50 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 10000 * torch.sum((state[:, -self.obj_dof:-1]) ** 2) # the screwdriver should only rotate in z direction

        goal_cost = torch.sum((500 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # add a running cost
        goal_cost += torch.sum((1 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))

        # obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
        # obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
        # # obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
        # goal_orientation = tf.euler_angles_to_matrix(goal[-self.obj_rotational_dim:], convention='XYZ')
        # # goal_orientation = tf.matrix_to_rotation_6d(goal_orientation) # convert to 6d representation  
        # cos_angle = tf.so3_relative_angle(obj_orientation, goal_orientation.unsqueeze(0), cos_angle=True)
        # # terminal cost
        # goal_cost = torch.sum((1000 * (1 - cos_angle[-1]) ** 2))
        # # running cost 
        # goal_cost = goal_cost + 10*torch.sum((1 * (1 - cos_angle) ** 2))

        # # terminal cost
        # goal_cost = torch.sum((100 * (obj_orientation[-1] - goal_orientation) ** 2))
        # # running cost 
        # goal_cost = goal_cost + torch.sum((3 * (obj_orientation - goal_orientation) ** 2))

        # if self.optimize_force:
        #     force = xu[:, self.dx + 4 * self.num_fingers: self.dx + (4 + 3) * self.num_fingers]
        #     force = force.reshape(force.shape[0], self.num_fingers, 3)
        #     force_norm = torch.norm(force, dim=-1)
        #     force_norm = force_norm - 2 # desired maginitute
        #     force_cost = 1 * torch.sum(((force_norm < 0) * force_norm) ** 2)
        #     action_cost += force_cost

        # index_ee_locs = self._ee_locations_in_screwdriver(partial_to_full_state(state[:, :4*self.num_fingers], fingers=self.fingers),
        #                                             state[:, 4*self.num_fingers: 4*self.num_fingers + self.obj_dof],
        #                                             queried_fingers=['index'])
        # index_pos_loss = 1000 * torch.sum((index_ee_locs[:,0] - self.default_index_ee_loc_in_screwdriver.unsqueeze(0)) ** 2)
        # return smoothness_cost + action_cost + goal_cost + upright_cost + index_pos_loss
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
            grad_g[:, T_range, T_range, :4 * self.num_fingers] = grad_g_q[:, 1:]
            # is valve in state
            if self.dx == 4 * self.num_fingers + self.obj_dof:
                grad_g[:, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
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

    def _index_contact_region_constraint(self, xu, compute_grads=True, compute_hess=False):
        " constraint specifying that the index finger only has contact with the top of the screwdriver"
        N, T, d = xu.shape
        env_q = xu[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof].reshape(-1, self.obj_dof)
        # Retrieve pre-processed data
        ret_scene = self.data['index']
        index_contact_pts = ret_scene['closest_pt_world'].reshape(N, T + 1, 3)[:, 1:]
        h = self.index_contact_region_constr(index_contact_pts.reshape(-1, 3), env_q.reshape(-1, self.obj_dof)).reshape(N, -1)
        if compute_grads:
            grad_h = torch.zeros(N, 1, T, T, d, device=self.device)
            dh_dcontact, dh_denv_q = self.grad_index_contact_region_constr(index_contact_pts.reshape(-1, 3), env_q.reshape(-1, self.obj_dof))
            dcontact_dq = ret_scene['closest_pt_q_grad'].reshape(N, T+1, 3, 4 * self.num_fingers)[:, 1:]
            dh_dq = dh_dcontact.reshape(N, T, 1, 3) @ dcontact_dq.reshape(N, T, 3, 4 * self.num_fingers)

            T_range = torch.arange(T, device=xu.device)
            grad_h[:, :, T_range, T_range, :4 * self.num_fingers] = dh_dq.reshape(N, T, 1, 4 * self.num_fingers).transpose(1, 2)
            grad_h[:, :, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dh_denv_q.reshape(N, T, 1, self.obj_dof).transpose(1, 2)
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
        # if self.optimize_force:
        #     u = u[:, :, 4 * self.num_fingers: (4 + 3) * self.num_fingers].reshape(-1, 3 * self.num_fingers)
        # else:
        u = u[:, :, :4 * self.num_fingers].reshape(-1, 4 * self.num_fingers)

        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1] # contact normal is pointing out 
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        # compute constraint value
        h = self.friction_vel_constr(u,
                                 contact_normal.reshape(-1, 3),
                                 contact_jac.reshape(-1, 3, 4 * self.num_fingers)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal, dh_djac = self.grad_friction_vel_constr(u,
                                                                   contact_normal.reshape(-1, 3),
                                                                   contact_jac.reshape(-1, 3, 4 * self.num_fingers))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, :-1]

            dh = dh_dnormal.shape[1]
            dh_dq = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :4 * self.num_fingers] = dh_dq[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range_plus, T_range_minus, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx: self.dx + 4 * self.num_fingers] = dh_du.reshape(N, T, dh, 4 * self.num_fingers).transpose(1, 2)
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
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            if self.collision_checking:
                print(f"max index repulsive constraint: {torch.max(h_rep)}")
            if self.contact_region:
                print(f"max contact region constraint: {torch.max(h_con_region)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
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
        if self.collision_checking:
            h = torch.cat((h, h_rep), dim=1)  
        if self.contact_region:
            h = torch.cat((h, h_con_region), dim=1)
        if compute_grads:
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
            # grad_h = torch.cat((grad_h, 
            #                     # grad_h_step_size,
            #                     grad_h_sin), dim=1)
            # grad_h = torch.cat((grad_h, grad_h_vel), dim=1)
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
    
  
    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the screwdriver once"
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('XYZ', screwdriver_goal).as_matrix()
    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    if params['use_arm']:
        robot_dof = 4 * num_fingers + 7
    else:
        robot_dof = 4 * num_fingers


    start = state.reshape(robot_dof + 4).to(device=params['device'])
    # start = torch.cat((state.reshape(10), torch.zeros(1).to(state.device))).to(device=params['device'])
    if params['controller'] == 'csvgd':
        # index finger is used for stability
        if 'index' in params['fingers']:
            contact_fingers = params['fingers']
        else:
            contact_fingers = ['index'] + params['fingers']        
        pregrasp_problem = AllegroContactProblem(
            dx=robot_dof,
            du=robot_dof,
            start=start[:robot_dof + obj_dof],
            goal=None,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_type=params['object_type'],
            world_trans=env.world_trans,
            fingers=contact_fingers,
            obj_dof_code=params['obj_dof_code'],
            obj_joint_dim=1,
            fixed_obj=True,
            use_arm=params['use_arm'],
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 50
    else:
        raise ValueError('Invalid controller')
    
    
    start = env.get_state()['q'].reshape(robot_dof + 4).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:robot_dof])

    if params['visualize_plan']:
        if params['use_arm']:
            camera_params = "screwdriver_w_arm"
        else:
            camera_params = "screwdriver"
        traj_for_viz = best_traj[:, :pregrasp_problem.dx]
        tmp = start[robot_dof:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)    
        viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, 
                             pregrasp_problem.fingers, pregrasp_problem.obj_dof+1, 
                             camera_params=camera_params, use_arm=params['use_arm'])


    for x in best_traj[:, :robot_dof]:
        action = x.reshape(-1, robot_dof).to(device=env.device) # move the rest fingers
        if params['mode'] == 'hardware':
            set_state = env.get_state()['all_state'].to(device=env.device)
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
            sim_viz_env.set_pose(set_state)
            sim_viz_env.step(action)
        env.step(action)
        action_list.append(action)
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, robot_dof)[0], params['fingers'], use_arm=params['use_arm']))

    state = env.get_state()
    start = state.reshape(robot_dof + 4).to(device=params['device'])
    turn_problem_fingers = params['fingers']
    turn_problem_start = start[:robot_dof + obj_dof]
    turn_problem = AllegroScrewdriver(
        start=turn_problem_start,
        goal=params['screwdriver_goal'],
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        friction_coefficient=params['friction_coefficient'],
        world_trans=env.world_trans,
        fingers=turn_problem_fingers,
        optimize_force=params['optimize_force'],
        force_balance=params['force_balance'],
        collision_checking=params['collision_checking'],
        obj_gravity=params['obj_gravity'],
        contact_region=params['contact_region'],
        static_init=params['static_init'],
        use_arm = params['use_arm'],
        arm_stiffness = 500,
        finger_stiffness=params['kp']
    )
    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)

    actual_trajectory = []
    duration = 0

    fig = plt.figure()
    axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_fingers}{i+1}'), projection='3d') for i in range(num_fingers)}
    for finger in params['fingers']:
        axes[finger].set_title(finger)
        axes[finger].set_aspect('equal')
        axes[finger].set_xlabel('x', labelpad=20)
        axes[finger].set_ylabel('y', labelpad=20)
        axes[finger].set_zlabel('z', labelpad=20)
        axes[finger].set_xlim3d(-0.05, 0.1)
        axes[finger].set_ylim3d(-0.06, 0.04)
        axes[finger].set_zlim3d(1.32, 1.43)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []


    for finger in params['fingers']:
        ee = state2ee_pos(start[:robot_dof], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    num_fingers_to_plan = num_fingers
    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state.reshape(robot_dof + 4).to(device=params['device'])

        actual_trajectory.append(state[:, :robot_dof + obj_dof].squeeze(0).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start[:robot_dof + obj_dof])
        
        if torch.isnan(best_traj).any().item():
            env.reset()
            break
        #debug only
        # turn_problem.save_history(f'{fpath.resolve()}/op_traj.pkl')

        print(f"solve time: {time.time() - start_time}")
        planned_theta_traj = best_traj[:, robot_dof: robot_dof + obj_dof].detach().cpu().numpy()
        print(f"current theta: {state[0, -(obj_dof+1): -1].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        # if k < params['num_steps'] - 1:
        #     if params['mode'] == 'hardware':
        #         pass # debug TODO: fix it
        #         # add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
        #     else:
        #         add_trajectories(trajectories, best_traj, axes, env, sim=sim, gym=gym, viewer=viewer,
        #                         config=params, state2ee_pos_func=state2ee_pos)

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :turn_problem.dx]
            traj_for_viz = torch.cat((start[:turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
        
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.viz_contact_scenes, viz_fpath, turn_problem.fingers, 
                                 turn_problem.obj_dof+1, camera_params=camera_params, use_arm=params['use_arm'])
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['optimize_force']:
            print("planned force")
            print(action[:, robot_dof:].reshape(num_fingers_to_plan, 3)) # print out the action for debugging
            # print("delta action")
            # print(action[:, :robot_dof].reshape(num_fingers_to_plan, 4))
        # print(action)
        action = action[:, :robot_dof]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers] # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            set_state = env.get_state()['all_state'].to(device=env.device)
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
            sim_viz_env.set_pose(set_state)
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers'], use_arm=params['use_arm']))
        # action = x[:, :4 * num_fingers].to(device=env.device)
        # NOTE: DEBUG ONLY
        # action = best_traj[1, :4 * turn_problem.num_fingers].unsqueeze(0)
        # action = action + torch.randn(action.shape).to(action.device) * 0.1
        env.step(action)
        action_list.append(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        screwdriver_state = env.get_state()['q'][:, -obj_dof-1: -1].cpu()
        screwdriver_mat = R.from_euler('XYZ', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(screwdriver_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()

        # distance2goal = (screwdriver_goal - screwdriver_state)).detach().cpu()
        if torch.isnan(distance2goal).item():
            env.reset()
            break
        print(distance2goal)
        if torch.isnan(distance2goal).any().item():
            env.reset()
            print("NAN")
            break
        info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        info_list.append(info)

        gym.clear_lines(viewer)
        state = env.get_state()
        start = state[:,:robot_dof + obj_dof].squeeze(0).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos(start[:robot_dof], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for finger in params['fingers']:
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    # action_list = torch.concat(action_list, dim=0)
    # with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
    #     pkl.dump(action_list, f)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    fig.tight_layout()
    fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    plt.savefig(f'{fpath.resolve()}/traj.png')
    plt.close()
    # plt.show()



    state = env.get_state()
    state = state.reshape(robot_dof + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[:robot_dof + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, robot_dof + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    screwdriver_state = actual_trajectory[:, -obj_dof:].cpu()
    screwdriver_mat = R.from_euler('XYZ', screwdriver_state).as_matrix()
    distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
        torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0],1,1), cos_angle=False).detach().cpu()

    final_distance_to_goal = torch.min(distance2goal.abs())

    print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    env.reset()
    return final_distance_to_goal.cpu().detach().item()

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from hardware.hardware_env import HardwareEnv
        # TODO, think about how to read that in simulator
        # default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
        #                             torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
        #                             torch.tensor([[0., 0.5, 0.0, 0.7]]).float(),
        #                             torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float()),
        #                             dim=1)
        default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                    torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                    dim=1)
        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj='screwdriver',
                          mode='relative',
                          gradual_control=True,
                          num_repeat=10)
        root_coor, root_ori = env.obj_reader.get_state()
        root_coor = root_coor / 1000 # convert to meters
        # robot_p = np.array([-0.025, -0.1, 1.33])
        robot_p = np.array([0, -0.095, 1.33])
        root_coor = root_coor + robot_p
        sim_env = RosAllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 friction_coefficient=1.0,
                                 device=config['sim_device'],
                                 valve=config['object_type'],
                                 video_save_path=img_save_dir,
                                 joint_stiffness=config['kp'],
                                 fingers=config['fingers'],
                                 table_pose=root_coor,
                                 )
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.table_pose = sim_env.table_pose
    else:
        env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                    use_cartesian_controller=False,
                                    viewer=True,
                                    steps_per_action=60,
                                    friction_coefficient=1.0,
                                    device=config['sim_device'],
                                    video_save_path=img_save_dir,
                                    joint_stiffness=config['kp'],
                                    fingers=config['fingers'],
                                    gradual_control=True,
                                    use_arm=config['use_arm'],
                                    )
        sim, gym, viewer = env.get_sim()
    if config['mode'] == 'hardware_copy':
        from hardware.hardware_env import RosNode
        ros_copy_node = RosNode()
        

    

    


    state = env.get_state()
    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass

    results = {}
    succ_rate = {}

    # set up the kinematic chain
    if config['use_arm']:
        asset = f'{get_assets_dir()}/xela_models/victor_allegro.urdf'
    else:
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    config['ee_names'] = ee_names
    config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans, use_arm=config['use_arm'])
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])


    for i in tqdm(range(config['num_trials'])):
        goal = - 90 / 180 * torch.tensor([0, 0, np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            succ = False
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['screwdriver_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node)
            if final_distance_to_goal < 30 / 180 * np.pi:
                succ = True
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
                succ_rate[controller] = [succ]
            else:
                results[controller].append(final_distance_to_goal)
                succ_rate[controller].append(succ)
        print(results)
        print(succ_rate)

    print(f"Average final distance to goal: {torch.mean(torch.tensor(results['csvgd']))}, std: {torch.std(torch.tensor(results['csvgd']))}")
    print(f"Success rate: {torch.mean(torch.tensor(succ_rate['csvgd']).float())}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


