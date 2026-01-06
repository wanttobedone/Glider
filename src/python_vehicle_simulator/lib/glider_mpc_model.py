#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glider_mpc_model.py:

    用CasADi符号变量重写Glider动力学方程。
    
    这是MPC的"预测模型"，求解器用它来预测未来状态轨迹。
    
    核心任务：
    - 把Glider.py中的dynamics()从Numpy翻译成CasADi
    - 提供RK4离散化积分器

Reference: Zhang et al. (2013) - Spiraling motion of underwater gliders

Author: Based on Glider.py dynamics
"""

import casadi as ca
import numpy as np
import math


class GliderMPCModel:
    """
    CasADi符号动力学模型
    
    使用方法：
        model = GliderMPCModel(glider)  # 从Glider对象提取参数
        F = model.get_integrator(Ts=0.5)  # 获取离散积分器
        x_next = F(x_k, u_k)  # 预测下一步状态
    """
    
    def __init__(self, glider=None):
        """
        初始化模型
        
        Args:
            glider: Glider对象，用于提取物理参数。如果None则使用默认参数。
        """
        # 提取物理参数
        self._load_parameters(glider)
        
        # 定义符号变量
        self._define_symbols()
        
        # 构建动力学方程
        self._build_dynamics()
        
        # 标记积分器是否已构建
        self._integrator_built = False
        self._integrator = None
        self._integrator_Ts = None
    
    def _load_parameters(self, glider):
        #从Glider对象加载物理参数，或使用默认值
        if glider is not None:
            # 从Glider对象提取参数
            self.rho = glider.rho
            self.g = glider.g
            
            # 静态块
            self.m_s = glider.m_s
            self.r_s = glider.r_s
            self.I_s = glider.I_s
            
            # 移动块
            self.m_r = glider.m_r
            self.R_r = glider.R_r
            self.I_r0 = glider.I_r0
            
            # 排水质量
            self.m = glider.m
            
            # 附加质量
            self.M_A = glider.M_A
            self.I_A = glider.I_A
            self.C_A = glider.C_A
            
            # 水动力系数
            self.K_D0 = glider.K_D0
            self.K_D = glider.K_D
            self.K_beta = glider.K_beta
            self.K_L0 = glider.K_L0
            self.K_alpha = glider.K_alpha
            self.K_MR = glider.K_MR
            self.K_p = glider.K_p
            self.K_M0 = glider.K_M0
            self.K_M = glider.K_M
            self.K_q = glider.K_q
            self.K_MY = glider.K_MY
            self.K_r = glider.K_r
            
            # 浮力中心位置
            self.r_b = glider.r_b
            
        else:
            # 使用Zhang论文的默认参数
            self.rho = 1025
            self.g = 9.81
            
            # Table 1 参数
            self.m_s = 54.28
            self.r_s = np.array([-0.0814, 0, 0.0032])
            self.I_s = np.diag([0.60, 15.27, 15.32])
            
            self.m_r = 11.0
            self.R_r = 0.014
            self.I_r0 = np.diag([0.02, 10.16, 0.17])
            
            self.m = 65.28
            
            # Table 3 参数
            self.M_A = np.diag([1.48, 49.58, 65.92])
            self.I_A = np.diag([0.53, 7.88, 10.18])
            self.C_A = np.array([
                [0, 0, 0],
                [0, 0, 3.61],
                [0, 2.57, 0]
            ])
            
            self.K_D0 = 7.19
            self.K_D = 386.29
            self.K_beta = -115.65
            self.K_L0 = -0.36
            self.K_alpha = 440.99
            self.K_MR = -58.27
            self.K_p = -19.83
            self.K_M0 = -0.28
            self.K_M = -65.84
            self.K_q = -205.64
            self.K_MY = -34.10
            self.K_r = -389.30
            
            self.r_b = np.array([0, 0, 0])
    
    def _define_symbols(self):
        # 定义CasADi符号变量
        # 状态向量 x (15维)
        # x = [eta(6), nu(6), xi(3)]
        self.x = ca.SX.sym('x', 15)
        
        # 控制向量 u (3维)
        # u = [r_rx_dot, gamma_dot, m_b_dot]
        self.u = ca.SX.sym('u', 3)
        
        # 拆解状态向量
        self.eta = self.x[0:6]      # 位置姿态 [x, y, z, phi, theta, psi]
        self.nu = self.x[6:12]      # 速度 [V1, V2, V3, p, q, r]
        self.xi = self.x[12:15]     # 执行器状态 [r_rx, gamma, m_b]
        
        # 进一步拆解
        self.pos = self.eta[0:3]    # [x, y, z]
        self.euler = self.eta[3:6]  # [phi, theta, psi]
        self.V = self.nu[0:3]       # [V1, V2, V3]
        self.Omega = self.nu[3:6]   # [p, q, r]
        
        self.r_rx = self.xi[0]
        self.gamma = self.xi[1]
        self.m_b = self.xi[2]
    
    def _build_dynamics(self):
        # 构建符号动力学方程 x_dot = f(x, u)，翻译自Glider.py的dynamics()函数
        # 提取状态分量
        phi = self.euler[0]
        theta = self.euler[1]
        psi = self.euler[2]
        
        V1 = self.V[0]
        V2 = self.V[1]
        V3 = self.V[2]
        p = self.Omega[0]
        q = self.Omega[1]
        r = self.Omega[2]
        
        r_rx = self.r_rx
        gamma = self.gamma
        m_b = self.m_b
        
        # 控制输入（速度指令）
        r_rx_dot = self.u[0]
        gamma_dot = self.u[1]
        m_b_dot = self.u[2]
        
        # 1. 计算移动块位置 r_r 和惯量 I_r
        r_r = self._compute_r_r(r_rx, gamma)
        I_r = self._compute_I_r(gamma)
        
        # 2. 构建广义惯性矩阵 M (6x6)
        # M_t = (m_r + m_s) * I_3 + M_A
        M_t = (self.m_r + self.m_s) * ca.SX.eye(3) + self._to_casadi_matrix(self.M_A)
        
        # C_t = C_A - m_s * skew(r_s) - m_r * skew(r_r)
        C_A_ca = self._to_casadi_matrix(self.C_A)
        r_s_ca = self._to_casadi_vector(self.r_s)
        C_t = C_A_ca - self.m_s * self._skew(r_s_ca) - self.m_r * self._skew(r_r)
        
        # I_t = I_s + I_r + I_A - m_r*skew(r_r)^2 - m_s*skew(r_s)^2
        I_s_ca = self._to_casadi_matrix(self.I_s)
        I_A_ca = self._to_casadi_matrix(self.I_A)
        skew_r_r = self._skew(r_r)
        skew_r_s = self._skew(r_s_ca)
        I_t = I_s_ca + I_r + I_A_ca - self.m_r * ca.mtimes(skew_r_r, skew_r_r) \
              - self.m_s * ca.mtimes(skew_r_s, skew_r_s)
        
        # 组装 6x6 惯性矩阵
        M = ca.SX.zeros(6, 6)
        M[0:3, 0:3] = M_t
        M[0:3, 3:6] = C_t
        M[3:6, 0:3] = C_t.T
        M[3:6, 3:6] = I_t
        
        # 3. 计算线动量 P 和角动量 Pi
        V_vec = ca.vertcat(V1, V2, V3)
        Omega_vec = ca.vertcat(p, q, r)
        
        # P = M_t * V - (m_s*skew(r_s) + m_r*skew(r_r)) * Omega
        P = ca.mtimes(M_t, V_vec) - ca.mtimes(
            self.m_s * skew_r_s + self.m_r * skew_r_r, Omega_vec)
        
        # Pi = (m_s*skew(r_s) + m_r*skew(r_r)) * V + I_t * Omega
        Pi = ca.mtimes(self.m_s * skew_r_s + self.m_r * skew_r_r, V_vec) \
             + ca.mtimes(I_t, Omega_vec)
        
        # 4. 计算水动力
        # 速度大小
        V_speed = ca.sqrt(V1**2 + V2**2 + V3**2 + 1e-6)  # 加小量防止除零
        V2_speed = V_speed**2
        
        # 攻角和侧滑角
        alpha = ca.atan2(V3, V1 + 1e-6)
        beta = ca.asin(ca.fmax(ca.fmin(V2 / V_speed, 1.0), -1.0))
        
        # 流体坐标系下的力
        D = (self.K_D0 + self.K_D * alpha**2) * V2_speed    # 阻力
        SF = self.K_beta * beta * V2_speed                   # 侧力
        L = (self.K_L0 + self.K_alpha * alpha) * V2_speed    # 升力
        
        # 流体坐标系下的力矩
        TDL1 = (self.K_MR * beta + self.K_p * p) * V2_speed      # 横滚力矩
        TDL2 = (self.K_M0 + self.K_M * alpha + self.K_q * q) * V2_speed  # 俯仰力矩
        TDL3 = (self.K_MY * beta + self.K_r * r) * V2_speed      # 偏航力矩
        
        # 流体力和力矩向量（流体坐标系）
        F_h = ca.vertcat(-D, SF, -L)
        T_h = ca.vertcat(TDL1, TDL2, TDL3)
        
        # 流体系到机体系的旋转矩阵
        R_bc = self._R_BC(alpha, beta)
        
        # 机体系下的水动力
        F = ca.mtimes(R_bc, F_h)
        T = ca.mtimes(R_bc, T_h)
    
        # 5. 计算重力/浮力项
        # 惯性系到机体系的旋转矩阵
        R_EB = self._Rzyx(phi, theta, psi)
        
        # 机体系下的重力方向向量
        k_body = ca.mtimes(R_EB.T, ca.vertcat(0, 0, 1))
        
        # 重力/浮力产生的力
        F_gravity = m_b * self.g * k_body
        
        # 重力/浮力产生的力矩
        r_b_ca = self._to_casadi_vector(self.r_b)
        r_gravity = self.m_r * r_r + self.m_s * r_s_ca + m_b * r_b_ca
        T_gravity = ca.cross(r_gravity, self.g * k_body)
        
        # 6. 计算科里奥利力/惯性力
        P_cross_Omega = ca.cross(P, Omega_vec)
        Pi_cross_Omega = ca.cross(Pi, Omega_vec)
        P_cross_V = ca.cross(P, V_vec)
        
        # 7. 组装右端项并求解加速度
        # 力方程右端
        rhs_force = P_cross_Omega + F_gravity + F
        
        # 力矩方程右端
        rhs_moment = Pi_cross_Omega + P_cross_V + T_gravity + T
        
        # 合并
        rhs = ca.vertcat(rhs_force, rhs_moment)
        
        # 求解 nu_dot = M^{-1} * rhs
        # 使用 ca.solve 而不是求逆，更稳定
        nu_dot = ca.solve(M, rhs)
        
        # 8. 计算运动学方程 eta_dot
        # 位置导数: pos_dot = R_EB * V
        pos_dot = ca.mtimes(R_EB, V_vec)
        
        # 姿态导数: euler_dot = T(phi, theta) * Omega
        T_euler = self._Tzyx(phi, theta)
        euler_dot = ca.mtimes(T_euler, Omega_vec)
        
        eta_dot = ca.vertcat(pos_dot, euler_dot)
        
        # 9. 执行器动力学 xi_dot = u
        xi_dot = self.u  # [r_rx_dot, gamma_dot, m_b_dot]
        
        # 10. 组装完整状态导数
        self.x_dot = ca.vertcat(eta_dot, nu_dot, xi_dot)
        
        # 创建CasADi函数
        self.f_dynamics = ca.Function('f_dynamics', [self.x, self.u], [self.x_dot],['x', 'u'], ['x_dot'])
    
    def _compute_r_r(self, r_rx, gamma):
        """
        计算移动质量块位置 (Eq. 9)
        r_r = [r_rx, R_r*cos(gamma + pi/2), R_r*sin(gamma + pi/2)]
            = [r_rx, -R_r*sin(gamma), R_r*cos(gamma)]
        """
        return ca.vertcat(
            r_rx,
            -self.R_r * ca.sin(gamma),
            self.R_r * ca.cos(gamma)
        )
    
    def _compute_I_r(self, gamma):
        """
        计算移动质量块惯量张量 (Eq. 12)
        I_r(gamma) = Rx(gamma)^T * I_r0 * Rx(gamma)
        """
        Rx = self._Rx(gamma)
        I_r0_ca = self._to_casadi_matrix(self.I_r0)
        return ca.mtimes(Rx.T, ca.mtimes(I_r0_ca, Rx))
    
    def _Rx(self, gamma):
        """绕x轴旋转矩阵"""
        cg = ca.cos(gamma)
        sg = ca.sin(gamma)
        return ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, cg, -sg),
            ca.horzcat(0, sg, cg)
        )
    
    def _R_BC(self, alpha, beta):
        """
        流体坐标系到机体坐标系的旋转矩阵 (Eq. 21)
        """
        ca_alpha = ca.cos(alpha)
        sa_alpha = ca.sin(alpha)
        cb = ca.cos(beta)
        sb = ca.sin(beta)
        
        return ca.vertcat(
            ca.horzcat(ca_alpha * cb, -ca_alpha * sb, -sa_alpha),
            ca.horzcat(sb, cb, 0),
            ca.horzcat(sa_alpha * cb, -sa_alpha * sb, ca_alpha)
        )
    
    def _Rzyx(self, phi, theta, psi):
        """
        欧拉角旋转矩阵 R_EB (ZYX顺序)
        从机体系到惯性系的旋转
        """
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)
        
        return ca.vertcat(
            ca.horzcat(cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*cphi*sth),
            ca.horzcat(spsi*cth, cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi),
            ca.horzcat(-sth, cth*sphi, cth*cphi)
        )
    
    def _Tzyx(self, phi, theta):
        """
        欧拉角速度转换矩阵
        euler_dot = T * Omega
        """
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        
        # 防止奇异（theta = +-90度）
        cth_safe = ca.if_else(ca.fabs(cth) < 1e-6, 1e-6, cth)
        
        return ca.vertcat(
            ca.horzcat(1, sphi*sth/cth_safe, cphi*sth/cth_safe),
            ca.horzcat(0, cphi, -sphi),
            ca.horzcat(0, sphi/cth_safe, cphi/cth_safe)
        )
    
    def _skew(self, v):
        """
        反对称矩阵 (skew-symmetric matrix)
        S(v) * w = v × w
        """
        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0)
        )
    
    def _to_casadi_matrix(self, np_matrix):
        """将numpy矩阵转换为CasADi DM/常量矩阵"""
        return ca.DM(np_matrix)
    
    def _to_casadi_vector(self, np_vector):
        """将numpy向量转换为CasADi DM/常量向量"""
        return ca.DM(np_vector)
    
    def build_integrator(self, Ts):
        """
        构建RK4离散化积分器
        
        Args:
            Ts: 采样周期 (s)
            
        Returns:
            F: CasADi Function，输入(x_k, u_k)，输出x_{k+1}
        """
        if self._integrator_built and self._integrator_Ts == Ts:
            return self._integrator
        
        # RK4积分
        k1 = self.f_dynamics(self.x, self.u)
        k2 = self.f_dynamics(self.x + Ts/2 * k1, self.u)
        k3 = self.f_dynamics(self.x + Ts/2 * k2, self.u)
        k4 = self.f_dynamics(self.x + Ts * k3, self.u)
        
        x_next = self.x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        self._integrator = ca.Function('F_rk4',
                                        [self.x, self.u],
                                        [x_next],
                                        ['x_k', 'u_k'], ['x_next'])
        self._integrator_built = True
        self._integrator_Ts = Ts
        
        return self._integrator
    
    def get_integrator(self, Ts):
        """获取积分器（如果不存在则构建）"""
        return self.build_integrator(Ts)
    
    def validate_against_numpy(self, glider, x0, u0, Ts=0.1, n_steps=10):
        """
        验证符号模型与Numpy模型的一致性
        
        Args:
            glider: Glider对象
            x0: 初始状态 (15,)
            u0: 控制输入 (3,) [r_rx_dot, gamma_dot, m_b_dot]
            Ts: 时间步长
            n_steps: 仿真步数
            
        Returns:
            errors: 每步的状态误差
        """
        F = self.get_integrator(Ts)
        
        # 符号模型轨迹
        x_casadi = x0.copy()
        casadi_trajectory = [x_casadi.copy()]
        
        # Numpy模型轨迹
        eta_np = x0[0:6].copy()
        nu_np = x0[6:12].copy()
        u_actual_np = x0[12:15].copy()  # [r_rx, gamma, m_b]
        numpy_trajectory = [np.concatenate([eta_np, nu_np, u_actual_np])]
        
        for step in range(n_steps):
            # CasADi模型推进
            x_casadi_next = np.array(F(x_casadi, u0)).flatten()
            casadi_trajectory.append(x_casadi_next.copy())
            x_casadi = x_casadi_next
            
            # Numpy模型推进
            # 注意：Glider.dynamics的控制输入是位置指令，需要转换
            # u_control = [r_rx_cmd, gamma_cmd, m_b_cmd]
            u_control_np = u_actual_np + u0 * Ts  # 积分得到位置指令
            nu_np, u_actual_np = glider.dynamics(eta_np, nu_np, u_actual_np, 
                                                  u_control_np, Ts)
            # 更新eta（这在原始代码中是在mainLoop里做的）
            from python_vehicle_simulator.lib.gnc import attitudeEuler
            eta_np = attitudeEuler(eta_np, nu_np, Ts)
            
            numpy_trajectory.append(np.concatenate([eta_np, nu_np, u_actual_np]))
        
        # 计算误差
        casadi_traj = np.array(casadi_trajectory)
        numpy_traj = np.array(numpy_trajectory)
        errors = np.abs(casadi_traj - numpy_traj)
        
        print("Validation Results:")
        print(f"  Max absolute error: {np.max(errors):.6e}")
        print(f"  Mean absolute error: {np.mean(errors):.6e}")
        print(f"  Error by state (max):")
        state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi', 
                      'V1', 'V2', 'V3', 'p', 'q', 'r',
                      'r_rx', 'gamma', 'm_b']
        for i, name in enumerate(state_names):
            print(f"    {name}: {np.max(errors[:, i]):.6e}")
        
        return errors, casadi_traj, numpy_traj

