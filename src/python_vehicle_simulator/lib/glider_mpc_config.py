#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glider_mpc_config.py:

    MPC控制器配置文件，所有超参数。
    - 时域参数（预测步数、采样周期）
    - 权重矩阵（Q, R, P）
    - 约束边界（状态约束、输入约束、执行器约束）
    - 软约束惩罚系数
    - 参考值（DIVE/CLIMB模式）

"""

import numpy as np
import math


class MPCConfig:
    """
    MPC超参数配置类
    
    使用方法：
        config = MPCConfig()
        config.N = 40  可以修改默认值
    """
    
    def __init__(self, z_max):
        
        # 时域参数
        self.N = 15                     # 预测步数
        self.Ts = 0.5                   # 采样周期 (s)
        self.T_horizon = self.N * self.Ts  # 预测时域 (s)
        
        # 状态向量定义 (15维)
        # x = [eta(6), nu(6), xi(3)]
        # eta = [x, y, z, phi, theta, psi]       位置和姿态
        # nu  = [V1, V2, V3, p, q, r]            线速度和角速度
        # xi  = [r_rx, gamma, m_b]               执行器状态
        self.n_states = 15
        self.n_controls = 3             # u = [r_rx_dot, gamma_dot, m_b_dot]
        
        # 状态索引（方便引用）
        self.idx_x = 0
        self.idx_y = 1
        self.idx_z = 2
        self.idx_phi = 3
        self.idx_theta = 4
        self.idx_psi = 5
        self.idx_V1 = 6
        self.idx_V2 = 7
        self.idx_V3 = 8
        self.idx_p = 9
        self.idx_q = 10
        self.idx_r = 11
        self.idx_r_rx = 12
        self.idx_gamma = 13
        self.idx_m_b = 14
        
        # 权重矩阵Q矩阵：状态误差权重 (15x15 对角)，重点跟踪俯仰角和前向速度，不关心位置过程，维持直线向前
        self.Q = np.diag([
            0.0,        # x - 不关心水平位置
            0.0,        # y - 不关心水平位置
            0.0,        # z - 不关心深度过程（由状态机管理终点）
            1000.0,      # phi - 保持水平（横滚）
            1000.0,     # theta - 【核心】俯仰角跟踪
            1000.0,       # psi - 航向保持
            100.0,       # V1 - 前向速度（维持滑翔）
            1000.0,       # V2 - 侧向速度（应该接近0）
            100.0,       # V3 - 垂向速度
            10000.0,     # p - 横滚角速度
            100.0,       # q - 俯仰角速度
            10000.0,     # r - 偏航角速度
            0.0,        # r_rx - 电池位置（低权重，让它自然调整）
            0.0,        # gamma - 电池旋转角
            0.0         # m_b - 净浮力
        ])
        
        # R矩阵：控制量权重 (3x3 对角)
        # 控制量是速度：[r_rx_dot, gamma_dot, m_b_dot]
        self.R = np.diag([
            10.0,       # r_rx_dot - 电池移动速度惩罚
            100000.0,       # gamma_dot - 电池旋转速度惩罚
            1.0         # m_b_dot - 浮力调节速度惩罚
        ])
        
        # P矩阵：终端代价权重 (15x15)
        # 通常取 P = alpha * Q，alpha > 1
        self.P = 10.0 * self.Q
        # 特别加大俯仰角的终端权重
        self.P[self.idx_theta, self.idx_theta] = 10000.0
        
        # 约束边界,输入约束,速度指令，硬约束
        self.r_rx_dot_max = 0.01        # 电池移动速度上限 (m/s)
        self.r_rx_dot_min = -0.01       # 电池移动速度下限 (m/s)
        self.gamma_dot_max = 0.05       # 电池旋转速度上限 (rad/s)
        self.gamma_dot_min = -0.05      # 电池旋转速度下限 (rad/s)
        self.m_b_dot_max = 0.1          # 浮力调节速度上限 (kg/s)
        self.m_b_dot_min = -0.1         # 浮力调节速度下限 (kg/s)
        
        # 打包成向量
        self.u_min = np.array([self.r_rx_dot_min, self.gamma_dot_min, self.m_b_dot_min])
        self.u_max = np.array([self.r_rx_dot_max, self.gamma_dot_max, self.m_b_dot_max])
        
        # 执行器约束（位置限制，硬约束）
        self.r_rx_min = 0.3516          # 电池位置下限 (m)
        self.r_rx_max = 0.4516          # 电池位置上限 (m)
        self.gamma_min = -math.pi / 2   # 电池旋转角下限 (rad)
        self.gamma_max = math.pi / 2    # 电池旋转角上限 (rad)
        self.m_b_min = -0.5             # 净浮力下限 (kg)
        self.m_b_max = 0.5              # 净浮力上限 (kg)
        
        # 状态约束（软约束）
        D2R = math.pi / 180
        self.theta_min = -50 * D2R      # 俯仰角下限 (rad)
        self.theta_max = 50 * D2R       # 俯仰角上限 (rad)
        self.phi_min = -30 * D2R        # 横滚角下限 (rad)
        self.phi_max = 30 * D2R         # 横滚角上限 (rad)
        self.V1_min = -0.5              # 前向速度下限 (m/s)，允许小幅倒退
        self.V1_max = 1.0               # 前向速度上限 (m/s)
        
        # 软约束状态的索引
        self.soft_constraint_indices = [
            self.idx_theta,
            self.idx_phi,
            self.idx_V1
        ]
        self.n_soft = len(self.soft_constraint_indices)
        
        # 软约束惩罚系数
        # J_soft = rho * epsilon^2 + mu * epsilon (L2 + L1)
        self.rho = 1e6                  # 二次惩罚系数
        self.mu = 1e4                   # 线性惩罚系数（防止微小违反）
        
        # 参考值设定
        D2R = math.pi / 180
        
        # ref设置，xyz完全不定义，向量中设置为0，后续路径规划更改
        # DIVE模式参考值
        self.dive_reference = {
            'theta': -35 * D2R,         # 目标俯仰角 (rad)
            'phi': 0.0,                 # 目标横滚角 (rad)
            'psi': 0.0,                 # 目标航向角 (rad)
            'V1': 0.3,                  # 目标前向速度 (m/s)
            'V2': 0.0,
            'V3': 0.0,
            'p': 0.0,
            'q': 0.0,
            'r': 0.0,
            'r_rx': 0.42,               # 电池略微靠前（低头）
            'gamma': 0.0,
            'm_b': 0.3                  # 正浮力质量 → 变重 → 下沉
        }
        
        # CLIMB模式参考值
        self.climb_reference = {
            'theta': 35 * D2R,          # 目标俯仰角 (rad)
            'phi': 0.0,
            'psi': 0.0,
            'V1': 0.3,
            'V2': 0.0,
            'V3': 0.0,
            'p': 0.0,
            'q': 0.0,
            'r': 0.0,
            'r_rx': 0.38,               # 电池略微靠后（抬头）
            'gamma': 0.0,
            'm_b': -0.3                 # 负浮力质量 → 变轻 → 上浮
        }
        
        # 锯齿运动深度范围
        self.z_min = 5.0                # 最浅深度 (m)
        self.z_max = z_max              # 最深深度 (m)
        
        # IPOPT求解器选项
        self.ipopt_options = {
            'ipopt.print_level': 0,                 # 静默输出
            'ipopt.max_iter': 100,                  # 最大迭代次数
            'ipopt.tol': 1e-4,                      # 收敛容差
            'ipopt.acceptable_tol': 1e-3,           # 可接受容差
            'ipopt.warm_start_init_point': 'yes',   # 启用热启动
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'print_time': False
        }
    
    def get_dive_reference(self):

        #获取DIVE模式的15维参考状态向量，Returns:   x_ref: (15,) numpy数组

        ref = self.dive_reference
        x_ref = np.zeros(self.n_states)
        # 位置不关心，设为0
        x_ref[self.idx_x] = 0.0
        x_ref[self.idx_y] = 0.0
        x_ref[self.idx_z] = 0.0
        # 姿态
        x_ref[self.idx_phi] = ref['phi']
        x_ref[self.idx_theta] = ref['theta']
        x_ref[self.idx_psi] = ref['psi']
        # 速度
        x_ref[self.idx_V1] = ref['V1']
        x_ref[self.idx_V2] = ref['V2']
        x_ref[self.idx_V3] = ref['V3']
        x_ref[self.idx_p] = ref['p']
        x_ref[self.idx_q] = ref['q']
        x_ref[self.idx_r] = ref['r']
        # 执行器
        x_ref[self.idx_r_rx] = ref['r_rx']
        x_ref[self.idx_gamma] = ref['gamma']
        x_ref[self.idx_m_b] = ref['m_b']
        
        return x_ref
    
    def get_climb_reference(self):

        #获取CLIMB模式的15维参考状态向量，Returns:   x_ref: (15,) numpy数组

        ref = self.climb_reference
        x_ref = np.zeros(self.n_states)
        x_ref[self.idx_x] = 0.0
        x_ref[self.idx_y] = 0.0
        x_ref[self.idx_z] = 0.0
        x_ref[self.idx_phi] = ref['phi']
        x_ref[self.idx_theta] = ref['theta']
        x_ref[self.idx_psi] = ref['psi']
        x_ref[self.idx_V1] = ref['V1']
        x_ref[self.idx_V2] = ref['V2']
        x_ref[self.idx_V3] = ref['V3']
        x_ref[self.idx_p] = ref['p']
        x_ref[self.idx_q] = ref['q']
        x_ref[self.idx_r] = ref['r']
        x_ref[self.idx_r_rx] = ref['r_rx']
        x_ref[self.idx_gamma] = ref['gamma']
        x_ref[self.idx_m_b] = ref['m_b']
        
        return x_ref
    
    def print_config(self):
        # 配置信息
        print("MPC Configuration")
        print(f"预测视野: N = {self.N}, Ts = {self.Ts}s, T = {self.T_horizon}s")
        print(f"状态维度: {self.n_states}")
        print(f"控制维度: {self.n_controls}")
        print("控制边界设置：")
        print(f"  r_rx_dot: [{self.r_rx_dot_min:.4f}, {self.r_rx_dot_max:.4f}] m/s")
        print(f"  gamma_dot: [{self.gamma_dot_min:.4f}, {self.gamma_dot_max:.4f}] rad/s")
        print(f"  m_b_dot: [{self.m_b_dot_min:.4f}, {self.m_b_dot_max:.4f}] kg/s")
        print("执行器边界:")
        print(f"  r_rx: [{self.r_rx_min:.4f}, {self.r_rx_max:.4f}] m")
        print(f"  gamma: [{math.degrees(self.gamma_min):.1f}, {math.degrees(self.gamma_max):.1f}] deg")
        print(f"  m_b: [{self.m_b_min:.2f}, {self.m_b_max:.2f}] kg")
        print("参考值设置:")
        print(f"  theta_ref: {math.degrees(self.dive_reference['theta']):.1f} deg")
        print(f"  V1_ref: {self.dive_reference['V1']:.2f} m/s")

