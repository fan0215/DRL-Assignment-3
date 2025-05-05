import random
import cv2
import os
import gc
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT # 确保动作空间匹配
from nes_py.wrappers import JoypadSpace # 评估脚本通常需要用这个包装环境
from collections import deque, namedtuple
import time
from tqdm import tqdm # tqdm 在 agent 中通常不需要，但保留引入
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Noisy Linear Layer (保持不变) ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Reset parameters
        self.reset_parameters()

        # Register buffers for noise (主要用于训练，评估时使用 mu)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        # 注意：评估时不需要 reset_noise()，所以这里初始化一次即可
        self.reset_noise()

    def reset_parameters(self):
        """Initialize the parameters"""
        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features)) # 注意：分母是 out_features

    def _scale_noise(self, size):
        """Generate factorized Gaussian noise"""
        # 使用与训练时相同的设备
        x = torch.randn(size, device=self.weight_mu.device)
        # 使用 .sgn() 而不是 .sign() 以兼容旧版本 PyTorch (如果需要)
        # 但 torch.sign() 是标准做法
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Reset the factorized noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out) # 直接使用输出噪声

    def forward(self, x):
        """Forward pass"""
        # 关键：在评估模式 (model.eval()) 或 非训练模式下，我们使用确定性的 mu 值
        # Noisy Nets for Exploration 论文建议在评估时移除噪声
        if not self.training: # 或者直接始终使用 mu 进行评估
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            # 训练时才加入噪声
            # 注意：训练代码不在此处，但这是 NoisyLinear 的完整实现方式
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # 每次训练迭代前需要调用 reset_noise() 来采样新的噪声

        return F.linear(x, weight, bias)


# --- Dueling CNN architecture (保持不变) ---
class DuelingCNN(nn.Module):
    def __init__(self, in_channels, num_actions, sigma_init=0.5):
        super(DuelingCNN, self).__init__()

        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate feature size dynamically
        # 使用 with torch.no_grad() 避免不必要的梯度计算
        with torch.no_grad():
            # 输入尺寸应与预处理后的帧尺寸匹配 (C, H, W) -> (4, 84, 90)
            dummy_input = torch.zeros(1, in_channels, 84, 90)
            feature_size = self.conv_layers(dummy_input).shape[1]
            # print(f"Calculated feature size: {feature_size}") # 调试用

        # Value stream (state value V(s))
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma_init) # 输出单一值 V(s)
        )

        # Advantage stream (action advantage A(s,a))
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma_init) # 输出每个动作的优势 A(s,a)
        )

    def forward(self, x):
        """Forward pass combining value and advantage streams"""
        # 确保输入是 float 类型，通常在转换成 Tensor 时处理
        # x = x.float() / 255.0 # 如果输入是 uint8 [0, 255]，则需要归一化
                               # 但 Agent 的 preprocess_frame 已经做了

        features = self.conv_layers(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 使用 advantage.mean(dim=1, keepdim=True) 在动作维度上计算均值
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def reset_noise(self):
        """Reset noise for all noisy layers"""
        # 这个方法主要在训练循环的每次迭代开始时调用
        # 评估时不需要调用
        for module in self.value_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- Agent Class (主要优化模型加载部分) ---
# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts using a loaded DQN model."""
    def __init__(self):
        # --- 基本设置 (保持不变) ---
        self.action_space = gym.spaces.Discrete(12) # 对应 COMPLEX_MOVEMENT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.frame_stack = 4
        # 使用 deque 来自动管理帧缓冲区的大小
        self.frame_buffer = deque(maxlen=self.frame_stack)
        # 初始化缓冲区为零帧 (float32 类型以匹配网络输入)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(np.zeros((84, 90), dtype=np.float32))

        self.skip_frames = 3 # 每次模型决策后，重复执行该动作 skip_frames 次
        self.skip_count = 0 # 当前还需要跳过的帧数
        self.last_action = 0 # 上一次选择的动作 (用于跳帧)

        self.step_counter = 0 # 用于可能的定期操作 (如 gc.collect)

        # --- 模型初始化 ---
        # 注意：这里的 num_actions=12 需要与 self.action_space.n 匹配
        self.model = DuelingCNN(self.frame_stack, self.action_space.n).to(self.device)

        # --- 优化的模型加载逻辑 ---
        model_loaded = False
        # 定义可能的模型文件路径列表
        model_paths = [
            'models/rainbow_icm_best.pth', # 原始路径
            'rainbow_icm_best.pth',      # 尝试当前目录
            'best_model.pth',           # 常见的备用名称
            'mario_dueling_dqn.pth'    # 另一个可能的名称
            # 如有需要，可添加更多可能的路径或名称
        ]

        print("--- Attempting to load pre-trained model ---")
        for path in model_paths:
            # 检查文件是否存在可以避免不必要的异常处理
            if os.path.exists(path):
                try:
                    print(f"Attempting to load model from: {path}")
                    # 使用 map_location 确保模型能加载到正确的设备 (CPU 或 GPU)
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                    print(f"--- Model loaded successfully from {path} ---")
                    model_loaded = True
                    break # 成功加载后即退出循环
                except Exception as e:
                    # 如果文件存在但加载失败，打印错误信息
                    print(f"Error loading model from {path}: {e}")
                    print("Ensure the model file is compatible with the current architecture.")
            else:
                # 如果文件路径不存在，告知用户
                print(f"Model file not found at: {path}")

        # --- 关键：检查模型是否成功加载 ---
        if not model_loaded:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("CRITICAL WARNING: Failed to load any pre-trained model.")
            print("The agent will use an UNTRAINED network.")
            print("Performance will be very poor (low score).")
            print("Please ensure a valid model file (e.g., 'models/rainbow_icm_best.pth')")
            print("exists in one of the expected locations relative to")
            print("where this script is being executed.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            # 这里可以选择是否抛出错误停止执行，取决于评估框架的要求
            # raise RuntimeError("Mandatory model file could not be loaded.")

        # --- 设置模型为评估模式 ---
        # 这会关闭 Dropout 等，并影响 NoisyLinear 的 forward 行为 (使用 mu)
        self.model.eval()
        print("Model set to evaluation mode.")

    def preprocess_frame(self, frame):
        """Convert RGB frame to grayscale, resize, and normalize."""
        # 输入 frame 期望是 H x W x C (例如 240, 256, 3) 的 NumPy 数组
        try:
            # 1. 检查输入是否为 NumPy 数组
            if not isinstance(frame, np.ndarray):
                # 如果环境返回的不是 numpy array (例如 LazyFrames)，先转换
                frame = np.array(frame)

            # 2. 转换为灰度图
            # 检查通道数，避免灰度图重复转换出错
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.ndim == 2:
                gray = frame # 已经是灰度图
            else:
                # 处理异常情况，例如意外的通道数
                print(f"Warning: Unexpected frame shape {frame.shape} in preprocess_frame. Using zeros.")
                return np.zeros((84, 90), dtype=np.float32)

            # 3. 调整大小 (宽 90, 高 84)
            # 使用 INTER_AREA 通常适合缩小图像
            resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)

            # 4. 归一化到 [0.0, 1.0] 并转换为 float32
            # 网络期望 float 类型输入
            normalized = resized.astype(np.float32) / 255.0

            return normalized
        except Exception as e:
            # 捕获并打印预处理中的任何错误
            print(f"Error in preprocess_frame: {e}")
            # 返回一个零数组，避免 Agent 崩溃，但可能影响性能
            return np.zeros((84, 90), dtype=np.float32)

    # Do not modify the input of the 'act' function.
    def act(self, observation):
        """Selects an action based on the observation."""
        try:
            self.step_counter += 1

            # --- Frame Skipping Logic ---
            # 如果 skip_count > 0，表示还在执行上一帧决定的动作
            if self.skip_count > 0:
                self.skip_count -= 1
                return self.last_action # 重复上一个动作

            # --- Process Observation and Get Action ---
            # 当 skip_count 为 0 时，需要处理新帧并做出新决策

            # 1. 预处理当前观测帧
            processed_frame = self.preprocess_frame(observation)

            # 2. 将处理后的帧添加到缓冲区
            self.frame_buffer.append(processed_frame)

            # 3. 将帧缓冲区堆叠成网络输入状态
            # deque 转 numpy array 会按添加顺序排列，(4, 84, 90)
            stacked_frames = np.array(self.frame_buffer)

            # 4. 转换为 PyTorch Tensor 并增加批次维度 (1, 4, 84, 90)
            # 发送到正确的设备 (CPU/GPU)
            state_tensor = torch.from_numpy(stacked_frames).unsqueeze(0).to(self.device)

            # 5. 使用模型进行推理 (在 torch.no_grad() 下进行评估以节省内存和计算)
            with torch.no_grad():
                q_values = self.model(state_tensor)

            # 6. 选择 Q 值最高的动作 (确定性策略)
            action = q_values.argmax(dim=1).item() # argmax 返回索引，item() 获取数值

            # --- Update State for Next Step ---
            # 7. 记录选择的动作，并重置跳帧计数器
            self.last_action = action
            # 下 skip_frames 帧将重复这个动作 (不包括当前帧，所以是 skip_frames - 1 次重复)
            # 但通常实现是决策一次，执行 N 次，所以这里设为 skip_frames
            # 第一次返回 action，之后 skip_frames-1 次在上面 if 分支返回 last_action
            # 如果 skip_frames=3，则决策1次，执行3次 (当前帧 + 跳过2帧) - 检查语义
            # 常见的 Frame Skipping (k=4): 决策 -> 执行动作 -> 环境 step 4 次 -> 获取最后状态
            # 这里的实现是：决策 -> 返回动作 -> (外部循环调用 act 3 次，都返回相同动作)
            # 如果 skip_frames=3，意味着模型每 3 帧决策一次。
            self.skip_count = self.skip_frames - 1 # 修正：决策后还需要跳过 skip_frames-1 帧

            # --- Optional: Periodic Garbage Collection ---
            # 避免内存泄漏或累积过多垃圾
            if self.step_counter % 100 == 0: # 每 100 步清理一次
                gc.collect()

            # 8. 返回选择的动作
            return action

        except Exception as e:
            # 捕获并打印 act 方法中的任何错误
            print(f"Error in Agent.act method: {e}")
            import traceback
            traceback.print_exc() # 打印详细的错误堆栈
            # 返回一个随机动作，以防评估框架期望总有返回值
            # 注意：这表示 Agent 出现了严重问题
            return self.action_space.sample()

# --- Optional: Test code ---
# 如果直接运行此文件，可以进行简单的测试
if __name__ == '__main__':
    # 实例化 Agent (会尝试加载模型)
    agent = Agent()

    # 创建一个虚拟的马里奥环境 (需要安装 gym-super-mario-bros)
    try:
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        # 使用与 Agent 动作空间匹配的 Wrapper
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    except Exception as e:
        print(f"Could not create gym environment: {e}")
        print("Please ensure 'gym-super-mario-bros' is installed.")
        env = None

    if env:
        print("Testing Agent in a dummy environment loop...")
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        max_steps = 500 # 限制测试步数

        while not done and step < max_steps:
            action = agent.act(obs)
            # 确保动作在有效范围内
            if not isinstance(action, int) or action < 0 or action >= agent.action_space.n:
               print(f"Warning: Agent returned invalid action: {action}. Sampling random action.")
               action = agent.action_space.sample()

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            env.render() # 显示游戏画面 (可能需要图形环境)
            time.sleep(0.01) # 稍微减慢速度以便观察

        print(f"Test finished after {step} steps.")
        print(f"Total reward received: {total_reward}")
        env.close()

    print("Agent structure and basic functionality test complete.")
    # 注意：这个测试不能完全替代官方的评估脚本，
    # 因为环境设置、Wrapper、评估指标可能不同。