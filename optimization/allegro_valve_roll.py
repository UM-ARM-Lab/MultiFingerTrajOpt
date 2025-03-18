# from isaacsim_hand_envs.allegro import AllegroValve # it needs to be imported before numpy and torch
from MFR_benchmark.utils import get_assets_dir

import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import torch
import time
import yaml
import copy
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import *

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat

def euler_to_angular_velocity(current_euler, next_euler):
    # using matrix
    
    current_mat = tf.euler_angles_to_matrix(current_euler, convention='XYZ')
    next_mat = tf.euler_angles_to_matrix(next_euler, convention='XYZ')
    dmat = next_mat - current_mat
    omega_mat = dmat @ current_mat.transpose(-1, -2)
    omega_x = (omega_mat[..., 2, 1] - omega_mat[..., 1, 2]) / 2
    omega_y = (omega_mat[..., 0, 2] - omega_mat[..., 2, 0]) / 2
    omega_z = (omega_mat[..., 1, 0] - omega_mat[..., 0, 1]) / 2
    omega = torch.stack((omega_x, omega_y, omega_z), dim=-1)

    # R.from_euler('XYZ', current_euler.cpu().detach().numpy().reshape(-1, 3)).as_quat().reshape(3, 12, 4)

    # quaternion 
    # current_quat = euler_to_quat(current_euler)
    # next_quat = euler_to_quat(next_euler)
    # dquat = next_quat - current_quat
    # con_quat = - current_quat # conjugate
    # con_quat[..., 0] = current_quat[..., 0]
    # omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:] 
    return omega

class PositionControlConstrainedSteinTrajOpt(ConstrainedSteinTrajOpt):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.torque_limit = params.get('torque_limit', 1)
        self.kp = params['kp']
        self.fingers = problem.fingers
        self.num_fingers = len(self.fingers)


    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            min_x = torch.cat((min_x, -1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)
            max_x = torch.cat((max_x, 1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

        if self.problem.du > 0:
            xuz_copy = xuz.reshape((N, self.problem.T, -1))
            robot_joint_angles = xuz_copy[:, :-1, :self.problem.robot_dof]
            robot_joint_angles = torch.cat(
                (self.problem.start[:self.problem.robot_dof].reshape((1, 1, self.problem.robot_dof)).repeat((N, 1, 1)), robot_joint_angles), dim=1)

            # make the commanded delta position respect the joint limits
            min_u_jlim = self.problem.robot_joint_x_min.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles
            max_u_jlim = self.problem.robot_joint_x_max.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles

            # make the commanded delta position respect the torque limits
            min_u_tlim = -self.torque_limit / self.kp * torch.ones_like(min_u_jlim)
            max_u_tlim = self.torque_limit / self.kp * torch.ones_like(max_u_jlim)

            # overall commanded delta position limits
            min_u = torch.where(min_u_jlim > min_u_tlim, min_u_jlim, min_u_tlim)
            max_u = torch.where(max_u_tlim > max_u_jlim, max_u_jlim, max_u_tlim)
            min_x = min_x.repeat((N, 1, 1)).to(device=xuz.device)
            max_x = max_x.repeat((N, 1, 1)).to(device=xuz.device)
            min_x[:, :, self.problem.dx:self.problem.dx + self.problem.robot_dof] = min_u
            max_x[:, :, self.problem.dx:self.problem.dx + self.problem.robot_dof] = max_u
            torch.clamp_(xuz, min=min_x.reshape((N, -1)), max=max_x.reshape((N, -1)))
    def resample(self, xuz):
        xuz = xuz.to(dtype=torch.float32)
        self.problem._preprocess(xuz)
        return super().resample(xuz)
class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)
    
class AllegroObjectProblem(ConstrainedSVGDProblem):

    def __init__(self, 
                 dx,
                 du,
                 start, # start should include both robot and obj states
                 goal, 
                 T, 
                 chain, 
                 world_trans, # transformation from the world to robot frame
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 obj_dof_code=[0, 0, 0, 0, 0, 0], 
                 obj_joint_dim=0,
                 fixed_obj=False,
                 device='cuda:0'):
        """
        obj_dof: DoF of the object, The max number is 6, It's the DoF for the rigid body, not including any joints within the object. 
        obj_joint_dim: It's the DoF of the joints within the object, excluding those are rigid body DoF.
        """
        super().__init__(start, goal, T, device)
        self.fixed_obj = fixed_obj
        self.dx, self.du = dx, du
        self.dg_per_t = 0
        self.dg_constant = 0
        self.device = device
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = False
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.obj_dof = np.sum(obj_dof_code)
        self.obj_translational_code = obj_dof_code[:3]
        self.obj_rotational_code = obj_dof_code[3:]
        self.obj_translational_dim = np.sum(self.obj_translational_code)
        self.obj_rotational_dim = np.sum(self.obj_rotational_code)
        self.obj_joint_dim = obj_joint_dim
        self.arm_dof = 0
        self.robot_dof = self.arm_dof + 4 * self.num_fingers
        self.chain = chain
        if self.fixed_obj:
            self.start_obj_pose = self.start[-self.obj_dof:]
            self.start = self.start[:self.robot_dof]


        self.chain = chain
        self.joint_index = {
            'index': list(np.array([0, 1, 2, 3]) + self.arm_dof),
            'middle': list(np.array([4, 5, 6, 7]) + self.arm_dof),
            'ring': list(np.array([8, 9, 10, 11]) + self.arm_dof),
            'thumb': list(np.array([12, 13, 14, 15]) + self.arm_dof)
        }
        self.all_joint_index = sum([self.joint_index[finger] for finger in self.fingers], [])
        self.collision_checking_ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        self.ee_names = {
            'index': 'hitosashi_ee',
            'middle': 'naka_ee',
            'ring': 'kusuri_ee',
            'thumb': 'oya_ee',
            }
        self.ee_link_idx = {finger: chain.frame_to_idx[ee_name] for finger, ee_name in self.ee_names.items()}
        self.frame_indices = torch.tensor([self.ee_link_idx[finger] for finger in self.fingers])

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)

        self.world_trans = world_trans.to(device=device)
        self.alpha = 10
        self.env_force = False
        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999]) + 0.05
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227]) - 0.05
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999]) + 0.05
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162]) - 0.05
        joint_min = {'index': index_x_min, 'middle': index_x_min, 'ring': index_x_min, 'thumb': thumb_x_min}
        joint_max = {'index': index_x_max, 'middle': index_x_max, 'ring': index_x_max, 'thumb': thumb_x_max}
        self.x_max = torch.cat([joint_max[finger] for finger in self.fingers])
        self.x_min = torch.cat([joint_min[finger] for finger in self.fingers])

        self.robot_joint_x_max = self.x_max.clone()
        self.robot_joint_x_min = self.x_min.clone()
        if self.du > 0:
            self.u_max = torch.ones(self.robot_dof) * np.pi / 5 
            self.u_min = - torch.ones(self.robot_dof) * np.pi / 5
            self.x_max = torch.cat((self.x_max, self.u_max))
            self.x_min = torch.cat((self.x_min, self.u_min))
        self.data = {}

        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

        self.singularity_constr = vmap(self._singularity_constr)
        self.grad_singularity_constr = vmap(jacrev(self._singularity_constr))

        self.grad_euler_to_angular_velocity = jacrev(euler_to_angular_velocity, argnums=(0,1))

        # DEBUG ONLY
        self.J_list = []
        self.contact_con = []
        self.force_con = []
        self.kinematics_con = []
        self.friction_con = []

        self.contact_con_mean = []
        self.force_con_mean = []
        self.kinematics_con_mean = []
        self.friction_con_mean = []
    def forward_kinematics(self, q):
        return self.chain.forward_kinematics(partial_to_full_state(q[:, :4*self.num_fingers], fingers=self.fingers), frame_indices=self.frame_indices)
    def save_history(self, save_dir):
        result_dict = {
            'J': self.J_list,
            'contact_con': self.contact_con,
            'force_con': self.force_con,
            'kinematics_con': self.kinematics_con,
            'friction_con': self.friction_con,
            'contact_con_mean': self.contact_con_mean,
            'force_con_mean': self.force_con_mean,
            'kinematics_con_mean': self.kinematics_con_mean,
            'friction_con_mean': self.friction_con_mean
        }
        with open(save_dir, 'wb') as f:
            pkl.dump(result_dict, f)

    
    def _cost(self, x, start, goal):
        raise NotImplementedError
    
    def _objective(self, x):
        x = x[:, :, :self.dx + self.du]
        N = x.shape[0]
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))
    def _step_size_limit(self, xu):
        N, T, _ = xu.shape
        u = xu[:, :, -self.du:]

        max_step_size = 0.1
        h_plus = u - max_step_size
        h_minus = -u - max_step_size
        h = torch.stack((h_plus, h_minus), dim=2)  # N x T x 2 x du

        grad_h = torch.zeros(N, T, 2, self.du, T, self.dx + self.du, device=xu.device)
        hess_h = torch.zeros(N, T * 2 * self.du, T * (self.dx + self.du), device=xu.device)
        # assign gradients
        T_range = torch.arange(0, T, device=xu.device)
        grad_h[:, T_range, 0, :, T_range, -self.du:] = torch.eye(self.du, device=xu.device)
        grad_h[:, T_range, 1, :, T_range, -self.du:] = -torch.eye(self.du, device=xu.device)

        return h.reshape(N, -1), grad_h.reshape(N, -1, T * (self.dx + self.du)), hess_h

    def _singularity_constr(self, contact_jac):
        # this will be vmapped
        A = contact_jac @ contact_jac.transpose(-1, -2)
        eig = torch.linalg.eigvals(A).abs()
        eig = torch.topk(eig, 2, dim=-1).values
        manipulability = eig[0] / eig[1] - 50
        return manipulability
        # manipulability = torch.sqrt(torch.prod(eig, dim=-1))
        # return 0.0001 - manipulability

    @all_finger_constraints
    def _singularity_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        q = xu[:, :, :self.robot_dof]
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:]
       
        # compute constraint value
        h = self.singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))
        h = h.reshape(N, -1)
        dh = 1
        # compute the gradient
        if compute_grads:
            dh_djac = self.grad_singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, 1:]

            dh_dq = dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :4 * self.num_fingers] = dh_dq[:, 1:].transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None
    
    @staticmethod
    def get_rotation_from_normal(normal_vector):
        """
        :param normal_vector: (batch_size, 3)
        :return: (batch_size, 3, 3) rotation matrix with normal vector as the z-axis
        """
        z_axis = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)
        # y_axis = torch.randn_like(z_axis)
        y_axis = torch.tensor([1.0, 1.0, 1.0], device=normal_vector.device)\
            .unsqueeze(0).repeat(normal_vector.shape[0], 1) / torch.sqrt(torch.tensor(3))
        y_axis = y_axis - torch.sum(y_axis * z_axis, dim=1).unsqueeze(-1) * z_axis
        y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)
        x_axis = torch.linalg.cross(y_axis, z_axis, dim=-1)
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)
        R = torch.stack((x_axis, y_axis, z_axis), dim=2)
        return R
    
    def _ee_locations_in_screwdriver(self, q_rob, q_env, queried_fingers, object_frame_name='screwdriver_body'):

        assert q_rob.shape[-1] == 16
        assert q_env.shape[-1] == self.obj_dof

        _q_env = q_env.clone()
        if self.obj_dof == 3:
            _q_env = torch.cat((q_env, torch.zeros_like(q_env[..., :1])), dim=-1)

        robot_trans = self.contact_scenes.robot_sdf.chain.forward_kinematics(q_rob.reshape(-1, 16))
        ee_locs = []

        for finger in queried_fingers:
            ee_locs.append(robot_trans[self.ee_names[finger]].get_matrix()[:, :3, -1])

        ee_locs = torch.stack(ee_locs, dim=1)

        # convert to scene base frame
        ee_locs = self.contact_scenes.scene_transform.inverse().transform_points(ee_locs)

        # convert to scene ee frame
        # Note, the FK here does not consider the change of the link center, specifically, it's the position of the joint connecting this link
        object_trans = self.contact_scenes.scene_sdf.chain.forward_kinematics(
            _q_env.reshape(-1, _q_env.shape[-1]))
        ee_locs = object_trans[object_frame_name].inverse().transform_points(ee_locs)

        return ee_locs
    
    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        # preprocess fingers
        self._preprocess(x)

        # compute objective
        J, grad_J, hess_J = self._objective(x)
        hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, self.T, -1)
        K = self.K(Xk, Xk, None)  # hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)  # @hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T * (self.dx + self.du))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        G, dG, hessG = self.combined_constraints(augmented_trajectory, compute_hess=self.compute_hess)
        # DEBUG ONLY
        # augmented_x = augmented_trajectory.reshape(N, self.T, self.dx + self.du + self.dz)
        # xu = augmented_x[:, :, :(self.dx + self.du)]
        # try:
        #     g = self._con_eq(xu, compute_grads=False, compute_hess=False, verbose=True)
        #     h = self._con_ineq(xu, compute_grads=False, compute_hess=False, verbose=True)
        #     self.J_list.append(J.mean().item())
        #     self.contact_con.append(g['contact'])
        #     self.force_con.append(g['force'])
        #     self.kinematics_con.append(g['kinematics'])
        #     self.friction_con.append(h['friction'])
        #     # self.friction_con.append(g['friction'])
        #     self.contact_con_mean.append(g['contact_mean'])
        #     self.force_con_mean.append(g['force_mean'])
        #     self.kinematics_con_mean.append(g['kinematics_mean'])
        #     self.friction_con_mean.append(h['friction_mean'])
        #     # self.friction_con_mean.append(g['friction_mean'])
        # except:
        #     pass
        
        if hessG is not None:
            hessG.detach_()

        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal

        # update functions that require start
        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics

        # DEBUG ONLY
        self.J_list = []
        self.contact_con = []
        self.force_con = []
        self.kinematics_con = []
        self.friction_con = []

        self.contact_con_mean = []
        self.force_con_mean = []
        self.kinematics_con_mean = []
        self.friction_con_mean = []

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        initialize with object not moving at all
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u[:, :, :self.arm_dof] = u[:, :, :self.arm_dof] * 0.1

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :self.robot_dof] + u[:, t, :self.robot_dof]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (self.robot_dof + self.obj_dof):
            theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu
    def check_validity(self, state):
        return True



class AllegroContactProblem(AllegroObjectProblem):
    
    def get_constraint_dim(self, T):
        self.dg_per_t = 0
        self.dg_constant = self.num_fingers
        self.dg = self.dg_per_t * T + self.dg_constant
        self.dz = 0  # one friction constraints per finger
        self.dh = self.dz * T  # inequality

    def __init__(self, 
                 dx,
                 du,
                 start, 
                 goal, 
                 T, 
                 chain, 
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 obj_dof_code=[0, 0, 0, 0, 0, 0], 
                 obj_joint_dim=0,
                 fixed_obj=False,
                 collision_checking=False,
                 device='cuda:0'):
        # object_location is different from object_asset_pos. object_asset_pos is 
        # used for pytorch volumetric. The asset of valve might contain something else such as a wall, a table
        # object_location is the location of the object joint, which is what we care for motion planning 
        super().__init__(dx=dx, du=du, start=start, goal=goal, T=T, chain=chain, 
                         world_trans=world_trans, fingers=fingers, obj_dof_code=obj_dof_code, 
                         obj_joint_dim=obj_joint_dim, fixed_obj=fixed_obj, device=device)
        self.collision_checking = collision_checking
        self.get_constraint_dim(T)

        # update x_max with valve angle
        obj_x_max = 10.0 * np.pi * torch.ones(self.obj_dof)
        obj_x_min = -10.0 * np.pi * torch.ones(self.obj_dof)
        if not fixed_obj:
            self.x_max = torch.cat((self.x_max[:self.robot_dof], obj_x_max, self.x_max[self.robot_dof:]))
            self.x_min = torch.cat((self.x_min[:self.robot_dof], obj_x_min, self.x_min[self.robot_dof:]))
        # add collision checking
        # collision check all of the non-finger tip links
        # collision_check_oya = ['allegro_hand_oya_finger_link_13',
        #                        'allegro_hand_oya_finger_link_14',
        #                        ]
        # collision_check_hitosashi = [
        #     'allegro_hand_hitosashi_finger_finger_link_2',
        #     'allegro_hand_hitosashi_finger_finger_link_1'
        # ]
        self.object_type = object_type
        if object_type == 'cuboid_valve':
            asset_object = get_assets_dir() + '/valve/valve_cuboid.urdf'
        elif object_type == 'cylinder_valve':
            asset_object = get_assets_dir() + '/valve/valve_cylinder.urdf'
        elif object_type == 'cross_valve':
            asset_object = get_assets_dir() + '/valve/valve_cross.urdf'
        elif object_type == 'screwdriver':
            asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
        elif object_type == 'screwdriver_6d':
            asset_object = get_assets_dir() + '/screwdriver/screwdriver_6d.urdf'
        elif object_type == 'screwdriver_translation':
            asset_object = get_assets_dir() + '/screwdriver/screwdriver_translation.urdf'
        elif object_type == 'peg':
            asset_object = get_assets_dir() + '/peg_insertion/peg.urdf'
        elif object_type == 'short_peg':
            asset_object = get_assets_dir() + '/peg_insertion/short_peg.urdf'
        elif object_type == 'batarang':
            asset_object = get_assets_dir() + '/reorientation/batarang.urdf'
        self.object_chain = pk.build_chain_from_urdf(open(asset_object).read()).to(device=self.device)
        self.object_asset_pos = torch.tensor(object_asset_pos).to(self.device).float()

        self._init_contact_scenes(asset_object, collision_checking)

    
    def _init_contact_scenes(self, asset_object, collision_checking):
        object_sdf = pv.RobotSDF(self.object_chain, path_prefix=None, use_collision_geometry=True) # since we are using primitive shapes for the object, there's no need to define path for stl
        robot_sdf = pv.RobotSDF(self.chain, path_prefix=get_assets_dir() + '/xela_models', use_collision_geometry=True)

        scene_trans = self.world_trans.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.object_asset_pos[0], self.object_asset_pos[1], self.object_asset_pos[2]))

        # self.index_collision_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
        #                                            collision_check_links=collision_check_hitosashi,
        #                                            softmin_temp=100.0)
        # self.thumb_collision_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
        #                                            collision_check_links=collision_check_oya,
        #                                            softmin_temp=100.0)
        # contact checking
        collision_check_links = [self.collision_checking_ee_names[finger] for finger in self.fingers]
        grad_smooth_points = 50
        if collision_checking:
            collision_check_links.append('allegro_hand_hitosashi_finger_finger_link_2')
            collision_check_links.append('allegro_hand_hitosashi_finger_finger_link_3')
        self.contact_scenes = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            partial_patch=False,
                                            grad_smooth_points=grad_smooth_points,
                                            )
        object_sdf = pv.RobotSDF(self.object_chain, path_prefix=None, use_collision_geometry=False) # since we are using primitive shapes for the object, there's no need to define path for stl
        robot_sdf = pv.RobotSDF(self.chain, path_prefix=get_assets_dir() + '/xela_models', use_collision_geometry=False)
        self.viz_contact_scenes = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                            collision_check_links=[self.collision_checking_ee_names['thumb']],
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            partial_patch=False,
                                            grad_smooth_points=grad_smooth_points,
                                            )
        # self.viz_contact_scenes.visualize_robot(partial_to_full_state(self.start[:self.robot_dof], fingers=self.fingers, arm_dof=self.arm_dof), None)

    def _preprocess(self, xu):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        x_expanded = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x_expanded[:, :, :self.robot_dof]
        if self.fixed_obj:
            theta = self.start_obj_pose.unsqueeze(0).repeat((N, self.T + 1, 1))
        else:
            theta = x_expanded[:, :, self.robot_dof: self.robot_dof + self.obj_dof]
        self._preprocess_fingers(q, theta)

    def _preprocess_fingers(self, q, theta):
        N, _, _ = q.shape

        # reshape to batch across time
        q_b = q.reshape(-1, self.robot_dof)
        theta_b = theta.reshape(-1, self.obj_dof)
        if self.obj_joint_dim > 0:
            theta_obj_joint = torch.zeros((theta_b.shape[0], self.obj_joint_dim),
                                          device=theta_b.device)  # add an additional dimension for the cap of the screw driver
            # the cap does not matter for the task, but needs to be included in the state for the model
            theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)
        full_q = partial_to_full_state(q_b, fingers=self.fingers, arm_dof=self.arm_dof)
        ret_scene = self.contact_scenes.scene_collision_check(full_q, theta_b,
                                                              compute_gradient=True,
                                                              compute_hessian=False)
        full_robot_dof = self.arm_dof + 16
        for i, finger in enumerate(self.fingers):
            self.data[finger] = {}
            self.data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, self.T + 1)
            # reshape and throw away data for unused fingers
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, self.T + 1, full_robot_dof)[:, :, self.all_joint_index]

            # contact jacobian
            contact_jacobian = ret_scene.get('contact_jacobian', None)
            self.data[finger]['contact_jacobian'] = contact_jacobian[:, i].reshape(N, self.T + 1, 3, full_robot_dof)[:, :, :, self.all_joint_index]

            # contact hessian
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_hessian = contact_hessian[:, i].reshape(N, self.T + 1, 3, full_robot_dof, full_robot_dof)[:, :, :, self.all_joint_index]
            contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            # contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # shape (N, T+1, 3, 8, 8)

            # gradient of contact point
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
            d_contact_loc_dq = d_contact_loc_dq[:, i].reshape(N, self.T + 1, 3, full_robot_dof)[:, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            self.data[finger]['closest_pt_q_grad'] = d_contact_loc_dq
            self.data[finger]['contact_hessian'] = contact_hessian
            self.data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][:, i] # the contact points are in the robot frame 
            self.data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

            # gradient of contact normal
            self.data[finger]['dnormal_dq'] = ret_scene['dnormal_dq'][:, i].reshape(N, self.T + 1, 3, full_robot_dof)[:, :, :, self.all_joint_index]  # [:, :, :,
            # self.all_joint_index]

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :self.obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :self.obj_dof]
            dJ_dq = contact_hessian
            self.data[finger]['dJ_dq'] = dJ_dq  # Jacobian of the contact point
        if self.collision_checking:
            self.data['allegro_hand_hitosashi_finger_finger_link_2'] = {}
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['sdf'] = ret_scene['sdf'][:, -2].reshape(N, self.T + 1)
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['grad_sdf'] = grad_g_q[:, -2].reshape(N, self.T + 1, full_robot_dof)[:, :, self.all_joint_index]
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, -2, :self.obj_dof]

            self.data['allegro_hand_hitosashi_finger_finger_link_3'] = {}
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['sdf'] = ret_scene['sdf'][:, -1].reshape(N, self.T + 1)
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['grad_sdf'] = grad_g_q[:, -1].reshape(N, self.T + 1, full_robot_dof)[:, :, self.all_joint_index]
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, -1, :self.obj_dof]


    @all_finger_constraints
    def _contact_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False, terminal=False):
        """
            Computes contact constraints
            constraint that sdf value is zero
        """
        N, T, _ = xu.shape
        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        g = ret_scene.get('sdf').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_q = ret_scene.get('grad_sdf', None)
        hess_g_q = ret_scene.get('hess_sdf', None)
        grad_g_theta = ret_scene.get('grad_env_sdf', None)
        hess_g_theta = ret_scene.get('hess_env_sdf', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)
        # g = g + 2e-3

        # If terminal, only consider last state
        if terminal:
            g = g[:, -1].reshape(N, 1)

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
            if terminal:
                grad_g = grad_g[:, -1].reshape(N, 1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
        
    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        action = xu[:, self.dx:]
        action_cost = torch.sum(action ** 2)
        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        # smoothness_cost += 1000 *  torch.sum((state[1:, :self.arm_dof] - state[:-1, :self.arm_dof]) ** 2) # penalize the arm movement
        smoothness_cost += 100 *  torch.sum((state[1:, :self.arm_dof] - state[:-1, :self.arm_dof]) ** 2) # penalize the arm movement
        return smoothness_cost + 10 * action_cost
    
    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess,
                                                                              terminal=True)
        return g_contact, grad_g_contact, hess_g_contact   
    def _con_ineq(self, x, compute_grads=True, compute_hess=False):
        return None, None, None

class AllegroValveTurning(AllegroContactProblem):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.screwdriver_force_balance:
            wrench_dim += 2

        self.dg_per_t = self.num_fingers * (1 + 2 + 4) + wrench_dim
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * self.num_fingers # one friction constraints per finger
        self.dz += self.num_fingers # minimum force constraint
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
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 finger_stiffness=3,
                 arm_stiffness=None,
                 obj_dof_code=[0, 0, 0, 0, 0, 0], 
                 obj_joint_dim=0,
                 screwdriver_force_balance=False,
                 collision_checking=False,
                 obj_gravity=False,
                 dx=None,
                 du=None,
                 contact_region=False,
                 device='cuda:0', **kwargs):
        self.screwdriver_force_balance = screwdriver_force_balance
        self.num_fingers = len(fingers)
        self.object_location = object_location
        self.obj_gravity = obj_gravity
        self.contact_region = contact_region
        self.finger_stiffness = finger_stiffness
        self.arm_stiffness = arm_stiffness
        self.obj_dof_code = obj_dof_code
        obj_dof = np.sum(obj_dof_code)
        self.arm_dof = 0
        self.robot_dof = self.arm_dof + 4 * self.num_fingers
        if dx is None:
            dx = self.robot_dof + obj_dof
        if du is None:
            du = self.robot_dof + 3 * self.num_fingers
        super().__init__(dx=dx, du=du, start=start, goal=goal, 
                         T=T, chain=chain, object_type=object_type, world_trans=world_trans,
                        object_asset_pos=object_asset_pos,
                         fingers=fingers, obj_dof_code=obj_dof_code, 
                         obj_joint_dim=obj_joint_dim, fixed_obj=False, 
                         collision_checking=collision_checking, device=device)
        self.friction_coefficient = friction_coefficient
        self.dynamics_constr = vmap(self._dynamics_constr)
        self.grad_dynamics_constr = vmap(jacrev(self._dynamics_constr, argnums=(0, 1, 2, 3, 4)))
        self.force_equlibrium_constr = vmap(self._force_equlibrium_constr_w_force)
        self.grad_force_equlibrium_constr = vmap(jacrev(self._force_equlibrium_constr_w_force, argnums=(0, 1, 2, 3, 4, 5, 6)))
        self.min_force_constr = vmap(self._min_force_constr, randomness='same')
        self.grad_min_force_constr = vmap(jacrev(self._min_force_constr, argnums=(0,)))

        max_f = torch.ones(3 * self.num_fingers) * 10
        min_f = torch.ones(3 * self.num_fingers) * -10
        self.x_max = torch.cat((self.x_max, max_f))
        self.x_min = torch.cat((self.x_min, min_f))
        self.min_force_dict = {'index': 0.1, 'middle': 0.1, 'ring': 0.1, 'thumb': 0.1}
        self.grad_min_force_constr = vmap(jacrev(self._min_force_constr, argnums=(0,)))
        self.friction_constr = vmap(self._friction_constr, randomness='same')
        self.grad_friction_constr = vmap(jacrev(self._friction_constr, argnums=(0, 1, 2)))

        self.kinematics_constr = vmap(vmap(self._kinematics_constr))
        self.grad_kinematics_constr = vmap(vmap(jacrev(self._kinematics_constr, argnums=(0, 1, 2, 3, 4, 5, 6))))

        self.kinematics_constr_w_proj = vmap(vmap(partial(self._kinematics_constr, projection=True)))
        self.grad_kinematics_constr_w_proj = vmap(vmap(jacrev(partial(self._kinematics_constr, projection=True), argnums=(0, 1, 2, 3, 4, 5, 6))))


    
    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 0.15 * torch.randn(N, self.T, 3 * self.num_fingers, device=self.device)
        u = torch.cat((u, force), dim=-1)
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

                theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
                theta = np.concatenate((theta_position, interp_rots), axis=-1)
                theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            else:
                theta = np.linspace(self.start[-self.obj_dof:].cpu().numpy(), self.goal.cpu().numpy(), self.T + 1)[1:]
                theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            theta = theta.unsqueeze(0).repeat((N,1,1))

            # DEBUG ONLY, use initial state as the initialization
            # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
            theta = torch.ones((N, self.T, self.obj_dof)).to(self.device) * self.start[-self.obj_dof:]
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu
    
    def _cost(self, xu, start, goal):
        # cost function for valve turning task
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + self.robot_dof]  # action dim = 8
        next_q = state[:-1, :-1] + action
        action_cost = 0
        smoothness_cost = 1 * torch.sum((state[1:] - state[:-1]) ** 2)
        smoothness_cost += 100 * torch.sum((state[1:, -1] - state[:-1, -1]) ** 2)

        goal_cost = (5000 * (state[-1, -1] - goal) ** 2).reshape(-1)
        # add a running cost
        goal_cost += torch.sum((10 * (state[:, -1] - goal) ** 2), dim=0)

        return smoothness_cost + action_cost + goal_cost

    def get_friction_polytope(self):
        """
        :param k: the number of faces of the friction cone
        :return: a list of normal vectors of the faces of the friction cone
        """
        normal_vectors = []
        for i in range(self.friction_polytope_k):
            theta = 2 * np.pi * i / self.friction_polytope_k
            # might be -cos(theta), -sin(theta), mu
            normal_vector = torch.tensor([np.cos(theta), np.sin(theta), self.friction_coefficient]).to(
                device=self.device,
                dtype=torch.float32)
            normal_vectors.append(normal_vector)
        normal_vectors = torch.stack(normal_vectors, dim=0)
        return normal_vectors
    
    def _force_equlibrium_constr_w_force(self, q, u, next_q, force_list, contact_jac_list, contact_point_list, next_env_q):
        # NOTE: the constriant is defined in the robot frame
        # NOTE: this only holds for quasi static system, as the reference point for the torque is essentially arbitrary
        # in a more general case, it has to be the CoM
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        delta_q = q + u - next_q
        torque_list = []
        reactional_torque_list = []
        for i, finger_name in enumerate(self.fingers):
            contact_jacobian = contact_jac_list[i]
            # transform everything into the robot frame
            force_robot_frame = self.world_trans.inverse().transform_normals(force_list[i].unsqueeze(0)).squeeze(0)
            reactional_torque_list.append(contact_jacobian.T @ -force_robot_frame)
            # pseudo inverse form
            contact_point_r_valve = contact_point_list[i] - obj_robot_frame[0]
            torque = torch.linalg.cross(contact_point_r_valve, force_robot_frame)
            torque_list.append(torque)
            # Force is in the robot frame instead of the world frame. 
            # It does not matter for comuputing the force equilibrium constraint
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        if self.obj_gravity:
            if self.obj_translational_dim > 0:
                g = self.obj_mass * torch.tensor([0, 0, -9.8], device=self.device, dtype=torch.float32)
                force_list = torch.cat((force_list, g.unsqueeze(0)))
                # force_list.append(g)
            if self.obj_rotational_dim > 0:
                if self.object_type == 'screwdriver':
                    # NOTE: only works for the screwdriver now
                    g = self.obj_mass * torch.tensor([0, 0, -9.8], device=self.device, dtype=torch.float32)
                    # add the additional dimension for the screwdriver cap
                    tmp = torch.zeros_like(next_env_q)
                    next_env_q = torch.cat((next_env_q, tmp[:1]), dim=-1)

                    body_tf = self.object_chain.forward_kinematics(next_env_q)['screwdriver_body']
                    body_com_pos = body_tf.get_matrix()[:, :3, -1]
                    torque = torch.linalg.cross(body_com_pos[0], g)
                    torque_list.append(torque)
        torque_list = torch.stack(torque_list, dim=0)
        sum_torque_list = torch.sum(torque_list, dim=0)
        sum_force_list = torch.sum(force_list, dim=0)

        g = []
        if self.obj_translational_dim > 0:
            g.append(sum_force_list)
        elif self.screwdriver_force_balance:
            # balance the force with the torque
            g.append(torch.sum(force_list, dim=0)[:2])
        if self.obj_rotational_dim > 0:
            g.append(sum_torque_list)
        g = torch.cat(g)
        reactional_torque_list = torch.stack(reactional_torque_list, dim=0)
        sum_reactional_torque = torch.sum(reactional_torque_list, dim=0)
        g_force_torque_balance = (sum_reactional_torque + self.finger_stiffness * delta_q)
        # print(g_force_torque_balance.max(), torque_list.max())
        g = torch.cat((g, g_force_torque_balance.reshape(-1)), dim=-1)
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g

    def _force_equlibrium_constraints_w_force(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        q = x[:, :-1, :self.robot_dof]
        next_q = x[:, 1:, :self.robot_dof]
        next_env_q = x[:, 1:, self.robot_dof:self.robot_dof + self.obj_dof]
        u = xu[:, :, self.dx: self.dx + self.robot_dof]
        force = xu[:, :, self.dx + self.robot_dof: self.dx + self.robot_dof + 3 * self.num_fingers]
        force_list = force.reshape((force.shape[0], force.shape[1], self.num_fingers, 3))
        # contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1].reshape(-1, 3, 4 * self.num_fingers)\
        #                      for finger_name in self.fingers]
        contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:].reshape(-1, 3, self.robot_dof)\
                             for finger_name in self.fingers]
        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=xu.device)
        contact_point_list = [self.data[finger_name]['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3) for finger_name in self.fingers]
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=xu.device)

        g = self.force_equlibrium_constr(q.reshape(-1, self.robot_dof), 
                                         u.reshape(-1, self.robot_dof), 
                                         next_q.reshape(-1, self.robot_dof), 
                                         force_list.reshape(-1, self.num_fingers, 3),
                                         contact_jac_list,
                                         contact_point_list,
                                         next_env_q.reshape(-1, self.obj_dof)).reshape(N, T, -1)
        # print(g.abs().max().detach().cpu().item(), g.abs().mean().detach().cpu().item())
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact, dg_dnext_env_q = self.grad_force_equlibrium_constr(q.reshape(-1, self.robot_dof), 
                                                                                  u.reshape(-1, self.robot_dof), 
                                                                                  next_q.reshape(-1, self.robot_dof), 
                                                                                  force_list.reshape(-1, self.num_fingers, 3),
                                                                                  contact_jac_list,
                                                                                  contact_point_list,
                                                                                  next_env_q.reshape(-1, self.obj_dof))
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], self.num_fingers * 3)
            
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            dg_dq = dg_dq.reshape(N, T, g.shape[2], self.robot_dof) 
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], self.robot_dof) 
            for i, finger_name in enumerate(self.fingers):
                # NOTE: assume fingers have joints independent of each other TODO: double check if this really requires independence
                # djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, :-1] # jacobian is the contact jacobian
                # dg_dq = dg_dq + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3,self.robot_dof, self.robot_dof)[:, 1:]
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
    
    def _kinematics_constr(self, current_q, next_q, current_theta, next_theta, contact_jac, contact_loc, contact_normal, projection=False):
        # approximate q dot and theta dot
        dq = next_q - current_q
        if np.sum(self.obj_rotational_code) == 3:
            obj_omega = euler_to_angular_velocity(current_theta[self.obj_translational_dim:],\
                                                   next_theta[self.obj_translational_dim:] )
        elif np.sum(self.obj_rotational_code) == 1:
            dtheta = next_theta[self.obj_translational_dim:] - current_theta[self.obj_translational_dim:]
            idx = torch.where(torch.tensor(self.obj_rotational_code) == 1)[0]
            obj_omega = torch.concat((torch.zeros_like(dtheta),
                                   torch.zeros_like(dtheta),
                                   torch.zeros_like(dtheta)), -1)  
            obj_omega[idx] = dtheta  
        contact_point_v = (contact_jac @ dq.reshape(self.robot_dof, 1)).squeeze(-1)  # should be N x T x 3
        # compute valve contact point velocity
        object_contact_point_v = 0 # in the robot frame
        if self.obj_rotational_dim > 0:
            if self.obj_translational_dim > 0:
                with torch.no_grad(): # the change of object location should not affect the contact velocity
                    obj_location = torch.zeros_like(current_q)[:3] # assume q has at least 3 dim
                    idx = torch.where(torch.tensor(self.obj_translational_code) == 1)[0]
                    obj_location[idx] = current_theta[:self.obj_translational_dim]
                    obj_location = obj_location + self.object_asset_pos
                    obj_robot_frame = self.world_trans.inverse().transform_points(obj_location.reshape(1, 3)).squeeze(0)
                contact_point_r_valve = contact_loc.reshape(3) - obj_robot_frame
            else: # assuming the object location is fixed
                obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3)).squeeze(0)
                contact_point_r_valve = contact_loc.reshape(3) - obj_robot_frame.reshape(3)
            obj_omega_robot_frame = self.world_trans.inverse().transform_normals(obj_omega.unsqueeze(0)).squeeze(0)
            object_contact_point_v = object_contact_point_v + torch.linalg.cross(obj_omega_robot_frame, contact_point_r_valve, dim=-1)
        if self.obj_translational_dim > 0:
            obj_translational_v =  torch.zeros_like(current_q)[:3] 
            idx = torch.where(torch.tensor(self.obj_translational_code) == 1)[0]
            obj_translational_v[idx] = next_theta[:self.obj_translational_dim] - current_theta[:self.obj_translational_dim]
            obj_translational_v = self.world_trans.inverse().transform_normals(obj_translational_v.reshape(1, 3)).squeeze(0)
            object_contact_point_v = object_contact_point_v + obj_translational_v
        # project the constraint into the tangential plane
        normal_projection = contact_normal.unsqueeze(-1) @ contact_normal.unsqueeze(-2)
        R = self.get_rotation_from_normal(contact_normal.reshape(-1, 3)).reshape(3, 3).detach().transpose(1,0)
        R = R[:2]
        # compute contact v tangential to surface
        # contact_point_v_tan = contact_point_v - (normal_projection @ contact_point_v.unsqueeze(-1)).squeeze(-1)
        # object_contact_point_v_tan = object_contact_point_v - (normal_projection @ object_contact_point_v.unsqueeze(-1)).squeeze(-1)

        # we actually ended up computing T+1 contact constraints, but start state is fixed so we throw that away
        # g = (contact_point_v - object_contact_point_v).reshape(N, -1) # DEBUG ONLY
        # g = (R @ (contact_point_v_tan - object_contact_point_v_tan).unsqueeze(-1)).squeeze(-1)
        if projection:
            g = (R @ (contact_point_v - object_contact_point_v).unsqueeze(-1)).squeeze(-1)
        else:
            g = contact_point_v - object_contact_point_v

        return g


    @all_finger_constraints
    def _kinematics_constraints(self, xu, finger_name, projection=False, compute_grads=True, compute_hess=False):
        """
            Computes on the kinematics of the valve and the finger being consistant
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        contact_jacobian = ret_scene.get('contact_jacobian', None)[:, :T+1]
        contact_loc = ret_scene.get('closest_pt_world', None).reshape(N, self.T + 1, 3)[:, :T + 1]
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)[:, :T+1]
        dJ_dq = ret_scene.get('dJ_dq', None)[:, :T+1]
        contact_normal = ret_scene.get('contact_normal', None).reshape(N, self.T + 1, 3)[:, :-1][:, :T]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, self.T + 1, 3, self.robot_dof)[:, :-1][:, :T]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, self.T + 1, 3, self.obj_dof)[:, :-1][:, :T]

        current_q = x[:, :-1, :self.robot_dof]
        next_q = x[:, 1:, :self.robot_dof]
        current_theta = x[:, :-1, self.robot_dof: self.robot_dof + self.obj_dof]
        next_theta = x[:, 1:, self.robot_dof: self.robot_dof + self.obj_dof]
        if projection:
            g = self.kinematics_constr_w_proj(current_q, next_q, current_theta, next_theta, contact_jacobian[:, :-1], contact_loc[:, :-1], contact_normal)
        else:
            g = self.kinematics_constr(current_q, next_q, current_theta, next_theta, contact_jacobian[:, :-1], contact_loc[:, :-1], contact_normal)
        g_dim = g.reshape(N, T, -1).shape[-1]
        g = g.reshape(N, -1)
        # print(finger_name, g.abs().max(), g.abs().mean())

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)

            if projection:
                dg_d_current_q, dg_d_next_q, dg_d_current_theta, dg_d_next_theta, dg_d_contact_jac, dg_d_contact_loc, dg_d_normal \
                = self.grad_kinematics_constr_w_proj(current_q, next_q, current_theta, next_theta, contact_jacobian[:, :-1], contact_loc[:, :-1], contact_normal)
            else:
                dg_d_current_q, dg_d_next_q, dg_d_current_theta, dg_d_next_theta, dg_d_contact_jac, dg_d_contact_loc, dg_d_normal \
                = self.grad_kinematics_constr(current_q, next_q, current_theta, next_theta, contact_jacobian[:, :-1], contact_loc[:, :-1], contact_normal)
            with torch.no_grad():

                dg_d_current_q = dg_d_current_q + dg_d_contact_jac.reshape(N, T, g_dim, -1) @ dJ_dq[:, :-1].reshape(N, T, -1, self.robot_dof)
                dg_d_current_q = dg_d_current_q + dg_d_contact_loc @ d_contact_loc_dq[:, :-1]
                # add constraints related to normal 
                dg_d_current_q = dg_d_current_q + dg_d_normal @ dnormal_dq
                dg_d_current_theta = dg_d_current_theta + dg_d_normal @ dnormal_dtheta

                grad_g = torch.zeros((N, T, T, g_dim, self.dx + self.du), device=x.device)
                grad_g[:, T_range_plus, T_range_minus, :, :self.robot_dof] = dg_d_current_q[:, 1:]
                grad_g[:, T_range_plus, T_range_minus, :, self.robot_dof: self.robot_dof + self.obj_dof] = dg_d_current_theta[:, 1:]
                grad_g[:, T_range, T_range, :, :self.robot_dof] = dg_d_next_q
                grad_g[:, T_range, T_range, :, self.robot_dof: self.robot_dof + self.obj_dof] = dg_d_next_theta
                grad_g = grad_g.permute(0, 1, 3, 2, 4).reshape(N, -1, T * (self.dx + self.du))
        
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
   
    
    def _dynamics_constr(self, q, u, next_q, contact_jacobian, contact_normal):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector
        dq = next_q - q
        contact_v = (contact_jacobian @ dq.unsqueeze(-1)).squeeze(-1)  # should be 3 vector
        # from commanded
        contact_v_u = (contact_jacobian @ u.unsqueeze(-1)).squeeze(-1)  # should be 3 vector

        # convert to world frame
        contact_v_world = self.world_trans.transform_normals(contact_v.unsqueeze(0)).squeeze(0)
        contact_v_u_world = self.world_trans.transform_normals(contact_v_u.unsqueeze(0)).squeeze(0)
        contact_normal_world = self.world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # compute projection onto normal
        normal_projection = contact_normal_world.unsqueeze(-1) @ contact_normal_world.unsqueeze(-2)

        # must find a lower dimensional representation of the constraint to avoid numerical issues
        # TODO for now hand coded, but need to find a better solution
        # R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).detach().permute(0, 1)
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).detach().permute(1, 0)
        R = R[:2]
        # compute contact v tangential to surface
        contact_v_tan = contact_v_world - (normal_projection @ contact_v_world.unsqueeze(-1)).squeeze(-1)
        contact_v_u_tan = contact_v_u_world - (normal_projection @ contact_v_u_world.unsqueeze(-1)).squeeze(-1)

        # should have same tangential components
        return (R @ (contact_v_tan - contact_v_u_tan).unsqueeze(-1)).squeeze(-1)
    
    @all_finger_constraints
    def _dynamics_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False):
        """ Computes dynamics constraints
            constraint that sdf value is zero
            also constraint on contact kinematics to get the valve dynamics
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x[:, :-1, :self.robot_dof]
        next_q = x[:, 1:, :self.robot_dof]
        u = xu[:, :, self.dx:]
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, self.robot_dof)[:, :-1]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        g = self.dynamics_constr(q.reshape(-1, self.robot_dof), u.reshape(-1, self.robot_dof),
                                 next_q.reshape(-1, self.robot_dof),
                                 contact_jac.reshape(-1, 3, self.robot_dof),
                                 contact_normal.reshape(-1, 3)).reshape(N, T, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, self.robot_dof, self.robot_dof)[:, :-1]
            dg_dq, dg_du, dg_dnext_q, dg_djac, dg_dnormal = self.grad_dynamics_constr(q.reshape(-1, self.robot_dof),
                                                                                      u.reshape(-1, self.robot_dof),
                                                                                      next_q.reshape(-1, self.robot_dof),
                                                                                      contact_jac.reshape(-1, 3, self.robot_dof),
                                                                                      contact_normal.reshape(-1, 3))

            dg_dq = dg_dq.reshape(N, T, g.shape[2], -1) + dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dq  #
            dg_dq = dg_dq + dg_djac.reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, self.robot_dof)
            dg_dtheta = dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dtheta

            grad_g[:, :, T_plus, T_minus, :self.robot_dof] = dg_dq[:, 1:].transpose(1, 2)  # first q is the start
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_plus, T_minus, self.robot_dof:self.robot_dof+self.obj_dof] = dg_dtheta[:, 1:].transpose(1, 2)
            grad_g[:, :, T_range, T_range, :self.robot_dof] = dg_dnext_q.reshape(N, T, -1, self.robot_dof).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None

        if compute_hess:
            hess_g = torch.zeros(N, T * 2,
                                 T * (self.dx + self.du),
                                 T * (self.dx + self.du), device=self.device)

            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess_g

        return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None
    
    def _friction_constr(self, dq, contact_normal, contact_jacobian):
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
        # here dq means the force in the world frame
        contact_v_contact_frame = R.transpose(0, 1) @ dq
        return B @ contact_v_contact_frame

    @all_finger_constraints
    def _friction_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        u = xu[:, :, self.dx:]
        u = u[:, :, self.robot_dof: (self.robot_dof + 3 * self.num_fingers)].reshape(-1, 3 * self.num_fingers)
        for i, finger_candidate in enumerate(self.fingers):
            if finger_candidate == finger_name:
                force_index = [i * 3 + j for j in range(3)]
                u = u[:, force_index]
                break
        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, 1:] # contact normal is pointing out 
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, self.obj_dof)[:, 1:]

        # compute constraint value
        h = self.friction_constr(u,
                                 contact_normal.reshape(-1, 3),
                                 contact_jac.reshape(-1, 3, self.robot_dof)).reshape(N, -1)
        if h.isnan().any():
            print("nan in friction constraint")
        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal, dh_djac = self.grad_friction_constr(u,
                                                                   contact_normal.reshape(-1, 3),
                                                                   contact_jac.reshape(-1, 3, self.robot_dof))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, self.robot_dof, self.robot_dof)[:, 1:]

            dh = dh_dnormal.shape[1]
            dh_dq = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, self.robot_dof)
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            # grad_h[:, :, T_range_plus, T_range_minus, :self.robot_dof] = dh_dq[:, 1:].transpose(1, 2)
            # grad_h[:, :, T_range_plus, T_range_minus, self.robot_dof: self.robot_dof + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)

            # friction constraint satisfied at the next time step
            grad_h[:, :, T_range, T_range, :self.robot_dof] = dh_dq[:, :].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.robot_dof: self.robot_dof + self.obj_dof] = dh_dtheta[:, :].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx + self.robot_dof + force_index[0]: self.dx + self.robot_dof + force_index[-1] + 1] = dh_du.reshape(N, T, dh, 3).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    def _min_force_constr(self, force_list):
        force_mag_offset = []
        # force_mag = []
        for i, finger_name in enumerate(self.fingers):
            min_force = self.min_force_dict[finger_name]
            force = force_list[i]
            force_norm = torch.linalg.norm(force, dim=-1)
            # force_mag.append(force / force_norm)
            force_mag_offset.append(min_force - force_norm)
            # print(force_norm.min(), min_force)
        # if self.env_force:
        #     # no constraint on the environment force
        #     force_mag_offset.append(torch.zeros_like(force_norm))
        return torch.stack(force_mag_offset, dim=0)
    
    def _min_force_constraints(self, xu, compute_grads=True, compute_hess=False, projected_diffusion=False):
        N, T, d = xu.shape
        device = xu.device
        # d = self.d
        force = xu[:, :, self.dx+self.robot_dof:]
        if self.env_force:
            num_forces = self.num_fingers + 1
        else:
            num_forces = self.num_fingers
        force_list = force.reshape(force.shape[0], force.shape[1], num_forces, 3)
        finger_force_list = force_list[:, :, :self.num_fingers]


        h = self.min_force_constr(finger_force_list.reshape(-1, self.num_fingers, 3),)
        h = h.reshape(N, T, -1)
        # dh_dforce = dh_dforce.reshape(N, T, -1, 3)
        if compute_grads:

            dh_dforce, = self.grad_min_force_constr(finger_force_list.reshape(-1, self.num_fingers, 3))
            dh_dforce = dh_dforce.reshape(dh_dforce.shape[0], dh_dforce.shape[1], self.num_fingers * 3)

            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)

            grad_g = torch.zeros(N, h.shape[2], T, T, d, device=self.device)

            mask_t = torch.zeros_like(grad_g).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_g).bool()
            mask_t_p[:, :, T_plus, T_minus] = True
            mask_force = torch.zeros_like(grad_g).bool()
            force_indices = torch.arange(self.dx + self.robot_dof, self.dx + self.robot_dof + self.num_fingers * 3, device=device)
            mask_force[:, :, :, :, force_indices] = True

            grad_g[torch.logical_and(mask_t, mask_force)] = dh_dforce.reshape(N, T, -1,
                                                                              self.num_fingers * 3
                                                                              ).transpose(1, 2).reshape(-1)
            grad_h = grad_g.transpose(1, 2)
            grad_h = grad_h.reshape(N, -1, T * d)

        else:
            return h.reshape(N, -1), None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * 3, T * 3, device=self.device)
            return h.reshape(N, -1), grad_h, hess_h
        return h.reshape(N, -1), grad_h, None
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
        
        # h_sin, grad_h_sin, hess_h_sin = self._singularity_constraint(
        #     xu=xu.reshape(-1, T, self.dx + self.du),
        #     compute_grads=compute_grads,
        #     compute_hess=compute_hess)
        
        # h_step_size, grad_h_step_size, hess_h_step_size = self._step_size_limit(xu)
        
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            print(f"max force constraint: {torch.max(h_force)}")
            # print(f"max force constraint: {torch.max(h_force)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
            # result_dict['singularity'] = torch.max(h_sin).item()
            return result_dict

        h = torch.cat((h,
                       h_force), dim=1)
        if compute_grads:
            grad_h = torch.cat((grad_h,
                            grad_h_force), dim=1)
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
            # grad_h = torch.cat((grad_h, 
            #                     # grad_h_step_size,
            #                     grad_h_sin), dim=1)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)
        # g_dynamics, grad_g_dynamics, hess_g_dynamics = self._dynamics_constraints(
        #     xu=xu.reshape(N, T, self.dx + self.du),
        #     compute_grads=compute_grads,
        #     compute_hess=compute_hess)
        g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints_w_force(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g_valve, grad_g_valve, hess_g_valve = self._kinematics_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            projection=True,
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        # g_valve_proj, grad_g_valve_proj, hess_g_valve_proj = self._kinematics_constraints(
        #     xu=xu.reshape(N, T, self.dx + self.du)[:, :1],
        #     projection=True,
        #     compute_grads=compute_grads,
        #     compute_hess=compute_hess)
        # g_valve = torch.cat((g_valve_proj, g_valve[:, 3 * self.num_fingers:]), dim=1)
        # if grad_g_valve is not None:
        #     target_shape = (grad_g_valve_proj.shape[0], grad_g_valve_proj.shape[1], grad_g_valve.shape[2])
        #     pad_size = target_shape[-1] - grad_g_valve_proj.size(-1)
        #     grad_g_valve_proj_padded = torch.nn.functional.pad(grad_g_valve_proj, (0, pad_size))
        #     grad_g_valve = torch.cat((grad_g_valve_proj_padded, grad_g_valve[:, 3 * self.num_fingers:]), dim=1)
        # if hess_g_valve is not None:
        #     hess_g_valve = torch.cat((hess_g_valve_proj, hess_g_valve[:, :, 3 * self.num_fingers:]), dim=1)
        

        if verbose:
            print(f"max contact constraint: {torch.max(torch.abs(g_contact))}")
            # print(f"max dynamics constraint: {torch.max(torch.abs(g_dynamics))}")
            print(f"max valve kinematics constraint: {torch.max(torch.abs(g_valve))}")
            print(f"max force equilibrium constraint: {torch.max(torch.abs(g_equil))}")
            result_dict = {}
            result_dict['contact'] = torch.max(torch.abs(g_contact)).item()
            # result_dict['dynamics'] = torch.max(torch.abs(g_dynamics)).item()
            result_dict['kinematics'] = torch.max(torch.abs(g_valve)).item()
            result_dict['force'] = torch.max(torch.abs(g_equil)).item()
            result_dict['contact_mean'] = torch.mean(torch.abs(g_contact)).item()
            # result_dict['dynamics_mean'] = torch.mean(torch.abs(g_dynamics)).item()
            result_dict['kinematics_mean'] = torch.mean(torch.abs(g_valve)).item()
            result_dict['force_mean'] = torch.mean(torch.abs(g_equil)).item()

            # DEBUG ONLY
            # result_dict['friction'] = torch.max(torch.abs(g_friction)).item()
            # result_dict['friction_mean'] = torch.mean(torch.abs(g_friction)).item()
            return result_dict
        g_contact = torch.cat((
                                g_contact, 
                            #    g_dynamics,
                               g_equil,
                               g_valve,
                            #    g_friction,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((
                                        grad_g_contact, 
                                        # grad_g_dynamics,
                                        grad_g_equil,
                                        grad_g_valve,
                                        # grad_g_friction,
                                        ), dim=1)
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((
                                        hess_g_contact, 
                                        # hess_g_dynamics,
                                        hess_g_equil,
                                        hess_g_valve,
                                        # hess_g_friction,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact
 