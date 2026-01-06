# 水下滑翔机运动仿真，ZigZag运动，PID，MPC

第11选项为滑翔机PID控制仿真；main中传入深度
第12选项为滑翔机MPC控制仿真；main中传入深度

## MPC

**架构**：
状态机外环控制爬升、下潜，内层MPC跟踪爬升、下潜任务（体现在参考值向量 `x_ref` 设计上，状态机管理两种状态的 `x_ref`），即跟踪 `x_ref` 设置的固定俯仰角、固定油囊浮力大小。

**状态向量 X (15维)**：
`x = [eta(6), nu(6), xi(3)]`

* `eta`：世界系位置、姿态（欧拉角）
* `nu`：机体系速度，角速度
* `xi`：机体内部执行机构位置 `r_rx` 电池包轴向位置，`gamma` 电池包绕x轴旋转角度，`m_b` 油囊静浮力质量

**控制向量 u (3维)**：

* `r_rx_dot`:电池移动速度
* `gamma_dot`:电池旋转速度
* `m_b_dot`:浮力调节速度

### 文件说明

* `\src\python_vehicle_simulator\lib\glider_mpc_config.py`

  * MPC参数配置；储存预测步数、采样周期
  * `X_ref`, `Q`, `R`, `P` 权重矩阵等
  * 约束边界、软约束乘法系数（用于防止特殊情况超出约束边界而导致无解）
  * 状态机
* `glider_mpc_model.py`

  * 动力学模型，用于作为转移函数；把动力学模型从 Numpy 翻译成 CasADi，可用自动微分
* `glider_mpc_solver.py`

  * MPC求解器封装，求解器使用 **IPOPT** 求解非线性二次规划

## PID

两个状态机切换上升下潜任务，只控制电池轴向位置和静浮力来控制运动

# 问题与后续改进

* NMPC计算太慢太慢；
* 目前由于目标只实现Zigzag运动，参考值设计非常粗糙，只用两个状态的状态机来管理两个x_ref，mpc只负责跟踪俯仰角与油囊，会导致到达深度目标后刹不住；此类问题后续确定MPC主要任务后来根据问题显式建模x_ref函数序列来实现不同的控制目标；
* 后续考虑加上一个简单的耗电模型来比较mpc与pid的能耗差异

MPC：\src\python_vehicle_simulator\lib\glider_mpc_config.pyMPC配置
