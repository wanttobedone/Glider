#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glider_mpc_solver.py:

    MPC求解器封装类。   
    功能：
    - 构建最优控制问题（OCP）
    - 管理约束（硬约束+软约束）
    - 求解非线性规划（NLP）
    - 热启动管理
    - 预测轨迹提取
"""

import casadi as ca
import numpy as np
import time


class GliderMPCSolver:
    """
    MPC求解器
    
    调用：
        solver = GliderMPCSolver(model, config)
        u_opt = solver.solve(x_current, x_ref)
    """
    
    def __init__(self, model, config):
        """
        初始化MPC求解器
         Args:
            model: GliderMPCModel实例
            config: MPCConfig实例
        """
        self.model = model
        self.config = config
        
        self.n_states = config.n_states #状态维度，15
        self.n_controls = config.n_controls #控制维度，3
        self.N = config.N                  #预测步数
        self.Ts = config.Ts                 #采样时间
        
        # 构建积分器，获取物理模型的离散化方程，x_dot=f积分，f是nu_dot
        self.F = model.get_integrator(self.Ts)
        
        # 构建OCP
        self._build_ocp() #调用优化问题构建方法，显式定义计算图
        
        # 初始化热启动缓存
        self._init_warm_start()
        
        # 存储上一次求解结果（用于可视化）
        self.last_X_opt = None
        self.last_U_opt = None
        self.solve_time = 0
        self.solve_success = False

         # 降采样计数器，仿真0.02s调用一次MPC，MPC采样时间0.5s，令MPC忽略过高的调用
        self.mpc_call_counter = 0
        self.mpc_call_interval = int(self.config.Ts / 0.02)  # 0.5/0.02 = 25
        self.last_u_control = np.array([
        self.config.dive_reference['r_rx'],
        self.config.dive_reference['gamma'],
        self.config.dive_reference['m_b']
    ])
    
    def _build_ocp(self):
        """
        构建最优控制问题（OCP）
        
        决策变量：
            X: 状态轨迹 (n_states × N+1)
            U: 控制序列 (n_controls × N)
            S: 松弛变量 (n_soft × N+1)，用于软约束
        """
        N = self.N
        nx = self.n_states
        nu = self.n_controls
        n_soft = self.config.n_soft
        
        # 定义决策变量
        # 状态轨迹
        self.X = ca.SX.sym('X', nx, N + 1)
        
        # 控制序列
        self.U = ca.SX.sym('U', nu, N)
        
        # 松弛变量（用于软约束）
        self.S = ca.SX.sym('S', n_soft, N + 1)
        
        # 定义参数（每次求解时传入的值）
        # 当前状态（初始条件）
        self.x_init = ca.SX.sym('x_init', nx)
        
        # 参考状态
        self.x_ref = ca.SX.sym('x_ref', nx)
        
        # 构建代价函数，初始化为零
        J = 0
        
        Q = ca.DM(self.config.Q)    #从config里传参
        R = ca.DM(self.config.R)
        P = ca.DM(self.config.P)
        
        # 阶段代价
        for k in range(N):
            # 状态误差
            x_err = self.X[:, k] - self.x_ref
            J += ca.mtimes([x_err.T, Q, x_err])
            
            # 控制代价
            J += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            
            # 软约束惩罚 (L2 + L1)
            J += self.config.rho * ca.dot(self.S[:, k], self.S[:, k])
            J += self.config.mu * ca.sum1(self.S[:, k])
        
        # 终端代价
        x_err_N = self.X[:, N] - self.x_ref
        J += ca.mtimes([x_err_N.T, P, x_err_N])
        
        # 终端软约束惩罚
        J += self.config.rho * ca.dot(self.S[:, N], self.S[:, N])
        J += self.config.mu * ca.sum1(self.S[:, N])
        
        self.J = J
        
        # 构建约束
        g = []      # 约束表达式
        lbg = []    # 约束下界
        ubg = []    # 约束上界
        
        # 初始状态约束（等式）
        g.append(self.X[:, 0] - self.x_init)    # 预测轨迹的第一步 X[:, 0] 必须等于当前的真实状态 x_init
        lbg += [0] * nx #减法结果上下界
        ubg += [0] * nx
        
        # 动力学约束（等式）
        for k in range(N):
            x_next = self.F(self.X[:, k], self.U[:, k])
            g.append(self.X[:, k + 1] - x_next)
            lbg += [0] * nx
            ubg += [0] * nx
        
        # 输入约束（硬约束）
        # u_min <= U[:, k] <= u_max
        # 这些在决策变量边界中处理，不在g中
        
        #执行器状态约束（硬约束）
        # 电池位置、旋转角、净浮力的物理限制
        for k in range(N + 1):
            # r_rx 约束 (索引12)
            g.append(self.X[self.config.idx_r_rx, k])
            lbg.append(self.config.r_rx_min)
            ubg.append(self.config.r_rx_max)
            
            # gamma 约束 (索引13)
            g.append(self.X[self.config.idx_gamma, k])
            lbg.append(self.config.gamma_min)
            ubg.append(self.config.gamma_max)
            
            # m_b 约束 (索引14)
            g.append(self.X[self.config.idx_m_b, k])
            lbg.append(self.config.m_b_min)
            ubg.append(self.config.m_b_max)
        
        # 状态软约束
        # x_min - S <= x <= x_max + S (等价于两个不等式)
        soft_bounds = [
            (self.config.idx_theta, self.config.theta_min, self.config.theta_max),
            (self.config.idx_phi, self.config.phi_min, self.config.phi_max),
            (self.config.idx_V1, self.config.V1_min, self.config.V1_max)
        ]
        
        for k in range(N + 1):
            for i, (idx, x_min, x_max) in enumerate(soft_bounds):
                # x >= x_min - S  =>  x + S >= x_min
                g.append(self.X[idx, k] + self.S[i, k])
                lbg.append(x_min)
                ubg.append(ca.inf)
                
                # x <= x_max + S  =>  x - S <= x_max
                g.append(self.X[idx, k] - self.S[i, k])
                lbg.append(-ca.inf)
                ubg.append(x_max)
        
        # 松弛变量非负约束
        # S >= 0 (在决策变量边界中处理)
        
        self.g = ca.vertcat(*g)
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)
        
        # 决策变量边界
        # 打包决策变量
        self.opt_vars = ca.vertcat(
            self.X.reshape((-1, 1)),
            self.U.reshape((-1, 1)),
            self.S.reshape((-1, 1))
        )
        
        n_X = nx * (N + 1)
        n_U = nu * N
        n_S = n_soft * (N + 1)
        self.n_opt_vars = n_X + n_U + n_S
        
        # 状态边界（宽松，主要约束在g中）
        lbx_X = -ca.inf * np.ones(n_X)
        ubx_X = ca.inf * np.ones(n_X)
        
        # 控制边界（硬约束）
        lbx_U = np.tile(self.config.u_min, N)
        ubx_U = np.tile(self.config.u_max, N)
        
        # 松弛变量边界（非负）
        lbx_S = np.zeros(n_S)
        ubx_S = ca.inf * np.ones(n_S)
        
        self.lbx = np.concatenate([lbx_X, lbx_U, lbx_S])
        self.ubx = np.concatenate([ubx_X, ubx_U, ubx_S])
        
        # 创建NLP求解器
        # 参数向量
        self.p = ca.vertcat(self.x_init, self.x_ref)
        
        nlp = {
            'x': self.opt_vars, #自变量
            'f': self.J,      #目标
            'g': self.g,    #约束
            'p': self.p #参数
        }
        
        # IPOPT选项
        opts = self.config.ipopt_options.copy()
        
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)   #实例化NLP求解器
        
        print(f"MPC Solver built: {self.n_opt_vars} variables, {len(self.lbg)} constraints")
    
    def _init_warm_start(self):
        """初始化热启动缓存"""
        N = self.N
        nx = self.n_states
        nu = self.n_controls
        n_soft = self.config.n_soft
        
        # 初始猜测：全零
        self.X_guess = np.zeros((nx, N + 1))
        self.U_guess = np.zeros((nu, N))
        self.S_guess = np.zeros((n_soft, N + 1))
        
        self.warm_start_initialized = False
    
    def _pack_guess(self):
        """将猜测值打包成向量"""
        return np.concatenate([
            self.X_guess.flatten('F'),  # 按列展开
            self.U_guess.flatten('F'),
            self.S_guess.flatten('F')
        ])
    
    def _unpack_solution(self, sol):
        """从解向量中提取X, U, S"""
        N = self.N
        nx = self.n_states
        nu = self.n_controls
        n_soft = self.config.n_soft
        
        x_opt = np.array(sol['x']).flatten()
        
        n_X = nx * (N + 1)
        n_U = nu * N
        
        X_opt = x_opt[:n_X].reshape((nx, N + 1), order='F')
        U_opt = x_opt[n_X:n_X + n_U].reshape((nu, N), order='F')
        S_opt = x_opt[n_X + n_U:].reshape((n_soft, N + 1), order='F')
        
        return X_opt, U_opt, S_opt
    
    def _shift_warm_start(self, X_opt, U_opt, S_opt):
        """
        移位策略更新热启动
        
        把序列左移一位，最后一位复制
        """
        # X: 左移，最后一列复制
        self.X_guess[:, :-1] = X_opt[:, 1:]
        self.X_guess[:, -1] = X_opt[:, -1]
        
        # U: 左移，最后一列复制
        self.U_guess[:, :-1] = U_opt[:, 1:]
        self.U_guess[:, -1] = U_opt[:, -1]
        
        # S: 左移，最后一列设为0（期望无约束违反）
        self.S_guess[:, :-1] = S_opt[:, 1:]
        self.S_guess[:, -1] = 0
        
        self.warm_start_initialized = True
    
    def solve(self, x_current, x_ref):
        """
        求解MPC问题
        
        Args:
            x_current: 当前状态 (15,)
            x_ref: 参考状态 (15,)
            
        Returns:
            u_opt: 最优控制量 (3,) [r_rx_dot, gamma_dot, m_b_dot]
        """
        start_time = time.time()
        
        # 如果是第一次求解或热启动未初始化，用当前状态初始化
        if not self.warm_start_initialized:
            for k in range(self.N + 1):
                self.X_guess[:, k] = x_current
        
        # 参数值
        p_val = np.concatenate([x_current, x_ref])
        
        # 初始猜测
        x0_val = self._pack_guess()
        
        # 求解
        try:
            sol = self.solver(
                x0=x0_val,  # 初始猜测
                lbx=self.lbx,   #决策变量下界
                ubx=self.ubx,   #决策变量上界
                lbg=self.lbg,   #约束下界
                ubg=self.ubg,   #约束上界
                p=p_val  #当前参数值
            )
            
            # 检查求解状态
            stats = self.solver.stats()
            self.solve_success = stats['success']
            
            if not self.solve_success:
                print(f"Warning: MPC solver did not converge. Return status: {stats['return_status']}")
            
            # 提取解
            X_opt, U_opt, S_opt = self._unpack_solution(sol)
            
            # 存储结果
            self.last_X_opt = X_opt
            self.last_U_opt = U_opt
            
            # 更新热启动
            self._shift_warm_start(X_opt, U_opt, S_opt)
            
            # 返回第一个控制量
            u_opt = U_opt[:, 0]
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            self.solve_success = False
            u_opt = np.zeros(self.n_controls)
        
        self.solve_time = time.time() - start_time
        
        return u_opt
    
    def get_predicted_trajectory(self):
        """
        获取上一次求解的预测轨迹
        
        Returns:
            X_opt: 预测状态轨迹 (15 × N+1)
            U_opt: 预测控制序列 (3 × N)
        """
        return self.last_X_opt, self.last_U_opt
    
    def get_solve_info(self):
        """
        获取求解信息
        
        Returns:
            dict: 包含求解时间、成功标志等
        """
        return {
            'solve_time': self.solve_time,
            'success': self.solve_success,
        }
    
    def reset_warm_start(self):
        """重置热启动（切换模式时调用）"""
        self.warm_start_initialized = False


class GliderMPCController:
    """
    完整的MPC控制器封装
    
    包含：
    - 状态机（DIVE/CLIMB切换）
    - MPC求解器
    - 控制量积分
    - 抗饱和处理
    
    使用方法：
        controller = GliderMPCController(glider)
        u_control = controller.compute_control(eta, nu, u_actual, sampleTime)
    """
    
    def __init__(self, glider=None):
        """
        初始化MPC控制器
        
        Args:
            glider: Glider对象（可选，用于提取物理参数）
        """
        from .glider_mpc_config import MPCConfig
        from .glider_mpc_model import GliderMPCModel

        # 获取目标深度如果传入了 glider 对象，就用它的 ref_z；否则默认 100
        target_z = glider.ref_z if glider is not None else 100.0
        
        # 将目标深度传给 MPCConfig
        self.config = MPCConfig(z_max=target_z)
        
        # 创建预测模型
        self.model = GliderMPCModel(glider)
        
        # 创建求解器
        self.solver = GliderMPCSolver(self.model, self.config)
        
        # 状态机
        self.flight_mode = "DIVE"
        
        # 内部积分状态（用于抗饱和）
        self.u_integrated = np.array([
            self.config.dive_reference['r_rx'],
            self.config.dive_reference['gamma'],
            self.config.dive_reference['m_b']
        ])
        
        print("GliderMPCController initialized.")
        self.config.print_config()

        # 降采样计数器
        self.mpc_call_counter = 0
        self.mpc_call_interval = int(self.config.Ts / 0.02)  # 0.5/0.02 = 25
        self.last_u_control = np.array([
            self.config.dive_reference['r_rx'],
            self.config.dive_reference['gamma'],
            self.config.dive_reference['m_b']
        ])
    
    def compute_control(self, eta, nu, u_actual, sampleTime):
        """
        计算MPC控制量，内外环，目前是外环状态机判断深度控制，内环MPC控制
        
        Args:
            eta: 位置姿态 (6,) [x, y, z, phi, theta, psi]
            nu: 速度 (6,) [V1, V2, V3, p, q, r]
            u_actual: 当前执行器状态 (3,) [r_rx, gamma, m_b]
            sampleTime: 采样时间
            
        Returns:
            u_control: 控制指令 (3,) [r_rx_cmd, gamma_cmd, m_b_cmd]
        """
        # 降采样，计算仿真时间与MPC计算时间比值，只有该调用的时候才真正调用MPC
        self.mpc_call_counter += 1
        if self.mpc_call_counter < self.mpc_call_interval:
            # 返回上一次的控制量（保持不变）
            return self.last_u_control
        
        self.mpc_call_counter = 0  # 重置计数器

        # 组装当前状态 (15维)
        x_current = np.concatenate([eta, nu, u_actual])
        
        # 状态机：根据深度切换模式
        z = eta[2]  # 深度
        
        if self.flight_mode == "DIVE":
            if z >= self.config.z_max:
                self.flight_mode = "CLIMB"
                self.solver.reset_warm_start()  # 切换模式时重置热启动
                print(f"Mode switch: DIVE -> CLIMB at z = {z:.1f}m")
            x_ref = self.config.get_dive_reference()
        else:  # CLIMB
            if z <= self.config.z_min:
                self.flight_mode = "DIVE"
                self.solver.reset_warm_start()
                print(f"Mode switch: CLIMB -> DIVE at z = {z:.1f}m")
            x_ref = self.config.get_climb_reference()
        
        # 调用MPC求解
        u_dot_opt = self.solver.solve(x_current, x_ref)
        # u_dot_opt = [r_rx_dot, gamma_dot, m_b_dot]
        
        # 积分得到位置指令（使用内部积分状态，避免噪声累积）
        self.u_integrated += u_dot_opt * sampleTime
        
        # 抗饱和：如果指令超限，截断并反向修正积分状态
        u_control = self.u_integrated.copy()
        
        # r_rx 饱和处理
        if u_control[0] > self.config.r_rx_max:
            u_control[0] = self.config.r_rx_max
            self.u_integrated[0] = self.config.r_rx_max
        elif u_control[0] < self.config.r_rx_min:
            u_control[0] = self.config.r_rx_min
            self.u_integrated[0] = self.config.r_rx_min
        
        # gamma 饱和处理
        if u_control[1] > self.config.gamma_max:
            u_control[1] = self.config.gamma_max
            self.u_integrated[1] = self.config.gamma_max
        elif u_control[1] < self.config.gamma_min:
            u_control[1] = self.config.gamma_min
            self.u_integrated[1] = self.config.gamma_min
        
        # m_b 饱和处理
        if u_control[2] > self.config.m_b_max:
            u_control[2] = self.config.m_b_max
            self.u_integrated[2] = self.config.m_b_max
        elif u_control[2] < self.config.m_b_min:
            u_control[2] = self.config.m_b_min
            self.u_integrated[2] = self.config.m_b_min
            
        self.last_u_control = u_control  # 保存结果，用于开头重复调用时候的控制量更新
        
        return u_control
    
    def get_flight_mode(self):
        """获取当前飞行模式"""
        return self.flight_mode
    
    def get_solve_info(self):
        """获取MPC求解信息"""
        return self.solver.get_solve_info()
    
    def get_predicted_trajectory(self):
        """获取预测轨迹"""
        return self.solver.get_predicted_trajectory()


