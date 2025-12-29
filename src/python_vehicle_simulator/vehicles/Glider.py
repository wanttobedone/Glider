#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glider.py:  

    坐标系要以几何中心为原点
    状态机要加一个停留时间
   Class for an underwater glider based on the Seawing glider model from:
   
   Zhang et al. (2013). "Spiraling motion of underwater gliders: Modeling, 
   analysis, and experimental results." Ocean Engineering 60 (2013) 1-13.
   
   The glider is controlled by:
   - A sliding movable mass block (battery) along x-axis for pitch control
   - A rotating movable mass block around x-axis for roll control (set to 0 for sawtooth motion)
   - Net buoyancy adjustment for dive/climb control
       
   glider()                           
       Step inputs for r_rx, gamma, and m_b
   
   glider('depthAutopilot', z_d)
       z_d: desired depth for sawtooth motion (m)

Methods:
        
    [nu, u_actual] = dynamics(eta, nu, u_actual, u_control, sampleTime) 
        Integrates the glider equations of motion using Euler's method.
        Control input: u_control = [r_rx, gamma, m_b]
            r_rx:  position of movable mass along x-axis (m)
            gamma: rotation angle of movable mass around x-axis (rad)
            m_b:   net buoyancy mass (kg)

    u = depthAutopilot(eta, nu, sampleTime) 
        PID controller for sawtooth gliding motion.
       
    u = stepInput(t) 
        Generates step inputs for testing.
       
References: 
    
    Zhang et al. (2013). Spiraling motion of underwater gliders: Modeling, 
        analysis, and experimental results. Ocean Engineering 60, 1-13.
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
        Control. 2nd Edition, Wiley.

"""

import numpy as np
import math
import sys
from python_vehicle_simulator.lib.control import PIDpolePlacement
from python_vehicle_simulator.lib.gnc import Rzyx, Smtrx, ssa


class Glider:
    """
    glider()
        Step inputs for movable mass position and net buoyancy
        
    glider('depthAutopilot', z_d) 
        Depth autopilot for sawtooth motion
        
    Inputs:
        z_d: desired depth for sawtooth motion (m), positive downwards
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r_z=0,
        V_current=0,
        beta_current=0,
    ):
        # Constants
        self.D2R = math.pi / 180        # deg2rad
        self.rho = 1025                 # density of water (kg/m^3)
        self.g = 9.81                   # acceleration of gravity (m/s^2)
        
        # 控制模式
        if controlSystem == "depthAutopilot":
            self.controlDescription = (
                "Depth autopilot for sawtooth motion, z_d = "
                + str(r_z) 
                + " m"
            )
        else:
            self.controlDescription = (
                "Step inputs for movable mass and net buoyancy"
            )
            controlSystem = "stepInput"
            
        self.ref_z = r_z
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R
        self.controlMode = controlSystem
        
        # 物理参数
        
        # Vehicle name and dimensions
        self.name = "Seawing Underwater Glider"
        self.L = 1.99                   # 长度 (m)
        self.D_glider = 0.22            # 直径 (m)
        
        # 静态块，以及机翼等
        self.m_s = 54.28                #质量kg
        self.r_s = np.array([-0.0814, 0, 0.0032], float)  # 质心位置m
        self.I_s = np.diag([0.60, 15.27, 15.32])          # 惯性张量(kg*m^2)
        
        # 电池包等移动块
        self.m_r = 11.0                 # mass
        self.R_r = 0.014                # 电池偏心半径（用于产生横滚 Roll 力矩 (m)现在先不用
        self.I_r0 = np.diag([0.02, 10.16, 0.17])  # 惯性张量(kg*m^2)
        
        # 限位
        self.r_rx_min = 0.3516          # 沿 X 轴滑动的机械限位（用于调节俯仰 Pitch）
        self.r_rx_max = 0.4516          # max 
        self.r_rx_nominal = 0.4016      # nominal position (m)
        
        # 滚转限制，不旋转就设置0
        self.gamma_min = -math.pi / 2   # min rotation angle (rad)
        self.gamma_max = math.pi / 2    # max
        
        # 静浮力
        self.m_b_min = -0.5             # min net buoyancy (kg)
        self.m_b_max = 0.5              # max net buoyancy (kg)
        self.r_b = np.array([0, 0, 0], float)  # 浮力中心位置，设置为机体系原点
        
        # Displaced fluid mass (when neutrally buoyant: m = m_s + m_r)
        self.m = 65.28                  # mass of displaced fluid (kg)
        

        # 水动力系数D，完全参照zhang的table3，用条带理论估算
        
        # 附加质量 M_A = diag([X_udot, Y_vdot, Z_wdot])，线运动，对应线加速度
        self.X_udot = 1.48              # kg
        self.Y_vdot = 49.58             # kg
        self.Z_wdot = 65.92             # kg
        self.M_A = np.diag([self.X_udot, self.Y_vdot, self.Z_wdot])
        
        # 角运动附加惯量：对应角加速度 I_A = diag([K_pdot, M_qdot, N_rdot])
        self.K_pdot = 0.53              # kg*m^2，绕纵轴旋转（Roll）
        self.M_qdot = 7.88              # kg*m^2，绕横轴（Pitch）
        self.N_rdot = 10.18             # kg*m^2，绕立轴（Yaw）
        self.I_A = np.diag([self.K_pdot, self.M_qdot, self.N_rdot])
        
        # 附加耦合项 C_A
        self.N_vdot = 2.57              # kg*m，绕 Z 轴偏航的力矩
        self.Z_qdot = 3.61              # kg*m，垂直方向的惯性力
        # C_A matrix (Eq. in Section 2.5)
        self.C_A = np.array([
            [0, 0, 0],
            [0, 0, self.Z_qdot],
            [0, self.N_vdot, 0]
        ], float)
        
        # Drag force coefficients: D = (K_D0 + K_D * alpha^2) * V^2
        self.K_D0 = 7.19                # kg/m，寄生阻力，代表直航时的摩擦力
        self.K_D = 386.29               # kg/m/rad^2，诱导阻力
        
        # Side force coefficient: SF = K_beta * beta * V^2
        self.K_beta = -115.65           # kg/m/rad 侧滑效应
        
        # Lift force coefficients: L = (K_L0 + K_alpha * alpha) * V^2
        self.K_L0 = -0.36               # kg/m 初始升力
        self.K_alpha = 440.99           # kg/m/rad 主升力
        
        # 力矩项：
        # 横滚力矩: TDL1 = (K_MR * beta + K_p * p) * V^2
        self.K_MR = -58.27              # kg/rad 横滚耦合
        self.K_p = -19.83               # kg*s/rad 横滚阻尼
        
        # 俯仰力矩: TDL2 = (K_M0 + K_M * alpha + K_q * q) * V^2
        self.K_M0 = -0.28               # kg 纵向稳定性
        self.K_M = -65.84               # kg/rad 纵向稳定性
        self.K_q = -205.64              # kg*s/rad^2 俯仰阻尼，q是抬头/低头的角速度
        
        # 偏航力矩: TDL3 = (K_MY * beta + K_r * r) * V^2
        self.K_MY = -34.10              # kg/rad 复原力矩
        self.K_r = -389.30              # kg*s/rad^2 偏航阻尼 r是转向角速度
        
        # 驱动器参数(随机设置的，后续根据实际更改)

        self.T_rx = 2.0                 # 一阶滤波器时间常数 (s)电池滑动
        self.T_gamma = 2.0              # 电池旋转的时间常数 (s)
        self.T_mb = 5.0                 # buoyancy adjustment time(s)
        
        # 状态初始化

        # 初始速度: nu = [V1, V2, V3, p, q, r] in body frame
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)
        
        # Actual actuator states: u_actual = [r_rx, gamma, m_b]，x位置，旋转角度，静浮力质量
        self.u_actual = np.array([self.r_rx_nominal, 0, 0], float) #当前时间，电池包和浮力状态
        
        # Control input names
        self.controls = [
            "Movable mass position r_rx (m)",
            "Movable mass rotation gamma (deg)",
            "Net buoyancy m_b (kg)"
        ]
        self.dimU = len(self.controls)  #控制输入的维数（这里是3）
        
        # Depth autopilot parameters
        
        # 锯齿运动参数
        self.z_max = 100.0              # maximum depth (m)
        self.z_min = 5.0                # minimum depth (m)
        self.flight_mode = "DIVE"       # initial mode: DIVE or CLIMB
        
        # Target pitch angles for gliding (from paper Fig. 12)
        self.theta_d_dive = -35.0 * self.D2R   # 下潜时的目标俯仰角 (rad)
        self.theta_d_climb = 35.0 * self.D2R   # 上升时的目标俯仰角 (rad)

        # Net buoyancy commands
        self.m_b_dive = 0.3             # 静吸油、排油指令 (kg)
        self.m_b_climb = -0.3         
        
        # PID controller parameters for pitch control
        self.e_theta_int = 0            # 积分误差
        self.wn_theta = 0.5             # 自然频率wn(rad/s)
        self.zeta_theta = 1.0           # 阻尼比
        
        # 俯仰角参考模型
        self.theta_d = 0                # 当前的理想角度
        self.q_d = 0                    # 当前的理想角速度
        self.a_d = 0                    # 当前的理想角加速度
        self.wn_d = 0.1                 # 自然频率
        self.zeta_d = 1.0               # 阻尼比
        self.q_max = 5.0 * self.D2R     # 最大角速度限制 (rad/s)

    # 坐标系转换旋转矩阵函数
    
    def Rx(self, gamma):
        """
        Rotation matrix around x-axis by angle gamma (Eq. 12)
        """
        cg = math.cos(gamma)
        sg = math.sin(gamma)
        return np.array([
            [1, 0, 0],
            [0, cg, -sg],
            [0, sg, cg]
        ], float)
    
    def R_BC(self, alpha, beta):
        """
        流体系到机体系的旋转矩阵 (Eq. 22)
        """
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cb = math.cos(beta)
        sb = math.sin(beta)
        
        return np.array([
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0],
            [sa * cb, -sa * sb, ca]
        ], float)
    
    def compute_r_r(self, r_rx, gamma):
        """
        移动质量块的位置计算
        r_r = [r_rx, R_r*cos(gamma + pi/2), R_r*sin(gamma + pi/2)]
            = [r_rx, -R_r*sin(gamma), R_r*cos(gamma)]
        r_r电池包质心相对于机体坐标系原点;r_rx沿x轴位置;gamma绕x轴旋转角度;R_r电池偏心半径
        """
        return np.array([
            r_rx,
            self.R_r * math.cos(gamma + math.pi/2),
            self.R_r * math.sin(gamma + math.pi/2)
        ], float)
    
    def compute_I_r(self, gamma):
        """
        移动质量块的惯量张量计算
        I_r(gamma) = Rx(gamma)^T * I_r0 * Rx(gamma)
        """
        Rx_gamma = self.Rx(gamma)
        return Rx_gamma.T @ self.I_r0 @ Rx_gamma
    

    #   Glider动力学方程
    
    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu, u_actual] = dynamics(eta, nu, u_actual, u_control, sampleTime)
        
        Integrates the glider equations of motion (Eq. 19) using Euler's method.
        
        Inputs:
            eta: position and attitude [x, y, z, phi, theta, psi] in inertial frame
            nu: velocity [V1, V2, V3, p, q, r] in body frame
            u_actual: actual actuator states [r_rx, gamma, m_b]
            u_control: commanded actuator states [r_rx_cmd, gamma_cmd, m_b_cmd]
            sampleTime: integration time step (s)
            
        Returns:
            nu: updated velocity vector
            u_actual: updated actuator states
        """
        
        # 状态值
        phi = eta[3]        # roll angle 惯性系
        theta = eta[4]      # pitch angle
        psi = eta[5]        # yaw angle
        
        V1 = nu[0]          # surge velocity 机体系
        V2 = nu[1]          # sway velocity
        V3 = nu[2]          # heave velocity
        p = nu[3]           # roll rate
        q = nu[4]           # pitch rate
        r = nu[5]           # yaw rate
        
        V = np.array([V1, V2, V3], float)
        Omega = np.array([p, q, r], float)
        
        # 执行器实际状态
        r_rx = u_actual[0]
        gamma = u_actual[1]
        m_b = u_actual[2]
        
        # Commanded actuator states
        r_rx_cmd = u_control[0]
        gamma_cmd = u_control[1]
        m_b_cmd = u_control[2]
        
        # 移动块的位置和惯量
        r_r = self.compute_r_r(r_rx, gamma)
        I_r = self.compute_I_r(gamma)
        
        # 广义惯性矩阵计算
        # M_t = (m_r + m_s) * I_3 + M_A
        M_t = (self.m_r + self.m_s) * np.eye(3) + self.M_A
        
        # C_t = C_A - m_s * skew(r_s) - m_r * skew(r_r)
        C_t = self.C_A - self.m_s * Smtrx(self.r_s) - self.m_r * Smtrx(r_r)
        
        # I_t = I_s + I_r(gamma) + I_A - m_r*skew(r_r)*skew(r_r) - m_s*skew(r_s)*skew(r_s)
        I_t = (self.I_s + I_r + self.I_A 
               - self.m_r * Smtrx(r_r) @ Smtrx(r_r) 
               - self.m_s * Smtrx(self.r_s) @ Smtrx(self.r_s))
        
        # Assemble 6x6 generalized inertia matrix
        M = np.zeros((6, 6))
        M[0:3, 0:3] = M_t   #线运动惯性子矩阵，静态块质量 + 移动质量块质量
        M[0:3, 3:6] = C_t   #耦合惯性子矩阵，附加质量的耦合矩阵
        M[3:6, 0:3] = C_t.T #转置
        M[3:6, 3:6] = I_t   #角运动惯性子矩阵，静态块+移动块的惯量张量 + 附加惯量 - 耦合项
        
        M_inv = np.linalg.inv(M)
        
        # 计算线动量P和角动量Pi
        # P = M_t * V - (m_s*skew(r_s) + m_r*skew(r_r)) * Omega
        P = M_t @ V - (self.m_s * Smtrx(self.r_s) + self.m_r * Smtrx(r_r)) @ Omega
        
        # Pi = (m_s*skew(r_s) + m_r*skew(r_r)) * V + I_t * Omega
        Pi = (self.m_s * Smtrx(self.r_s) + self.m_r * Smtrx(r_r)) @ V + I_t @ Omega
        
        # 计算流体力和动量
        # Speed
        V_speed = math.sqrt(V1**2 + V2**2 + V3**2)
        
        if V_speed < 0.001:
            # Avoid division by zero at very low speeds
            alpha = 0
            beta = 0
        else:
            # Attack angle (Eq. after Eq. 3)
            alpha = math.atan2(V3, V1) if abs(V1) > 1e-6 else 0
            # Slip angle
            beta = math.asin(np.clip(V2 / V_speed, -1, 1))
        
        V2_speed = V_speed ** 2
        
        # 流体坐标系下的流体力
        D = (self.K_D0 + self.K_D * alpha**2) * V2_speed       # Drag
        SF = self.K_beta * beta * V2_speed                      # Side force
        L = (self.K_L0 + self.K_alpha * alpha) * V2_speed       # Lift
        
        # 流体坐标系下的流体力矩
        TDL1 = (self.K_MR * beta + self.K_p * p) * V2_speed     # Roll moment
        TDL2 = (self.K_M0 + self.K_M * alpha + self.K_q * q) * V2_speed  # Pitch moment
        TDL3 = (self.K_MY * beta + self.K_r * r) * V2_speed     # Yaw moment
        
        # Forces and moments in flow frame
        F_h = np.array([-D, SF, -L], float)
        T_h = np.array([TDL1, TDL2, TDL3], float)
        
        # 转换到机体系
        R_bc = self.R_BC(alpha, beta)
        F = R_bc @ F_h
        T = R_bc @ T_h
        
        # 计算重力、浮力项
        # 惯性系到机体系的旋转矩阵
        R_EB = Rzyx(phi, theta, psi)
        
        # 机体系下重力向量: k_body = R_EB^T * [0, 0, 1]^T
        k_body = R_EB.T @ np.array([0, 0, 1], float)
        
        # 重力分量
        F_gravity = m_b * self.g * k_body
        
        # 重力力矩分量
        # 力臂: m_r*r_r + m_s*r_s + m_b*r_b
        r_gravity = self.m_r * r_r + self.m_s * self.r_s + m_b * self.r_b
        T_gravity = np.cross(r_gravity, self.g * k_body)
        
        # 科里奥利力
        # P x Omega
        P_cross_Omega = np.cross(P, Omega)
        
        # Pi x Omega + P x V
        Pi_cross_Omega = np.cross(Pi, Omega)
        P_cross_V = np.cross(P, V)
        
        # Assemble right-hand side of Eq. 19 (simplified, assuming M_dot ≈ 0)
        # M_dot质量变化率忽略
        
        # Force equation 
        rhs_force = P_cross_Omega + F_gravity + F #科里奥利力+静重力、浮力+流体动力
        
        # Moment equation 
        rhs_moment = Pi_cross_Omega + P_cross_V + T_gravity + T #惯性力矩+重力矩+流体动力矩
        
        # Combined right-hand side
        rhs = np.concatenate([rhs_force, rhs_moment])
        

        #求解加速度: mu_dot = M_inv * rhs
        nu_dot = M_inv @ rhs

        # 驱动器 一阶滞后模型
       
        r_rx_dot = (r_rx_cmd - r_rx) / self.T_rx
        gamma_dot = (gamma_cmd - gamma) / self.T_gamma
        m_b_dot = (m_b_cmd - m_b) / self.T_mb
        

        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        
        r_rx = r_rx + sampleTime * r_rx_dot
        gamma = gamma + sampleTime * gamma_dot
        m_b = m_b + sampleTime * m_b_dot
        
        # 控制饱和限制
        r_rx = np.clip(r_rx, self.r_rx_min, self.r_rx_max)
        gamma = np.clip(gamma, self.gamma_min, self.gamma_max)
        m_b = np.clip(m_b, self.m_b_min, self.m_b_max)
        
        u_actual = np.array([r_rx, gamma, m_b], float)
        
        return nu, u_actual

    # Step input for open-loop testing
    
    def stepInput(self, t):
        """
        u_control = stepInput(t) generates step inputs for testing.
        
        Returns:
            u_control = [r_rx, gamma, m_b]
        """
        
        # 下潜: move mass forward, increase buoyancy (become heavier)
        if t < 100:
            r_rx = 0.42         # forward position
            gamma = 0           # no rotation (sawtooth motion)
            m_b = 0.3           # positive = heavier than water = sink
        
        # 上浮: move mass backward, decrease buoyancy (become lighter)
        elif t < 200:
            r_rx = 0.38         # backward position
            gamma = 0
            m_b = -0.3          # negative = lighter than water = rise
        
        # Repeat dive
        else:
            r_rx = 0.42
            gamma = 0
            m_b = 0.3
        
        u_control = np.array([r_rx, gamma, m_b], float)
        
        return u_control

    # Depth autopilot
    
    def depthAutopilot(self, eta, nu, sampleTime):
        """
        u_control = depthAutopilot(eta, nu, sampleTime)
        
        PID 控制器实现锯齿式下潜运动，通过调节移动电池包位置和净浮力，上浮下潜两种模式由静浮力调节
        
        Returns:
            u_control = [r_rx, gamma, m_b]
        """
        
        # Extract states
        z = eta[2]              # depth (positive downward)
        theta = eta[4]          # pitch angle
        q = nu[4]               # pitch rate
        

        # 状态机: DIVE <-> CLIMB
        if self.flight_mode == "DIVE":
            if z >= self.ref_z:
                self.flight_mode = "CLIMB"
                self.e_theta_int = 0  # reset integral 防止积分饱和
                
        elif self.flight_mode == "CLIMB":
            if z <= self.z_min:
                self.flight_mode = "DIVE"
                self.e_theta_int = 0  # reset integral
        
        # 设定目标
        if self.flight_mode == "DIVE":
            theta_target = self.theta_d_dive    # -35度
            m_b_cmd = self.m_b_dive             # 变重
        else:
            theta_target = self.theta_d_climb   # +35度
            m_b_cmd = self.m_b_climb            # 变轻
        
        # Pitch PID controller
        # Error
        e_theta = ssa(theta - theta_target)
        e_q = q - 0  # 目标角速度是 0
        
        # PID 增益，通过极点配置法设计，现在简化为硬编码的
        # Using a simple P-D controller for pitch
        Kp = 0.5    # proportional gain
        Kd = 0.2    # derivative gain
        Ki = 0.01   # integral gain
        
        # PID output: correction to battery position
        # Positive error (pitch too high) -> move mass forward -> decrease r_rx
        # Negative error (pitch too low) -> move mass backward -> increase r_rx
        delta_r_rx = Kp * e_theta + Kd * e_q + Ki * self.e_theta_int
        
        # Nominal position + correction
        r_rx_cmd = self.r_rx_nominal + delta_r_rx
        
        # 限制幅度
        r_rx_cmd = np.clip(r_rx_cmd, self.r_rx_min, self.r_rx_max)
        
        # 积分状态更新
        self.e_theta_int += sampleTime * e_theta
        
        # Output
        gamma_cmd = 0   # 锯齿运动，电池包不旋转
        
        u_control = np.array([r_rx_cmd, gamma_cmd, m_b_cmd], float)
        
        return u_control