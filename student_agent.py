import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import cv2
import gc
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time


class NoisyLinear(nn.Module):
    """實現帶有參數噪聲的線性層，用於探索性強化學習"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # 可學習參數
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # 初始化參數
        self.reset_parameters()
        
        # 註冊噪聲緩衝區
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化參數"""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """生成因子化高斯噪聲"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """重置因子化噪聲"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """前向傳播 - 評估時只使用平均值"""
        weight = self.weight_mu
        bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingCNN(nn.Module):
    """雙重卷積神經網絡架構，分隔狀態值估計和優勢函數"""
    def __init__(self, in_channels, num_actions, sigma_init=0.5):
        super(DuelingCNN, self).__init__()
        
        # 特徵提取層 - 使用更高效的參數配置
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),  # inplace操作節省內存
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # 計算特徵大小
        self.feature_size = self._get_conv_output((in_channels, 84, 90))
        
        # 值流 (狀態價值 V(s))
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, sigma_init),
            nn.ReLU(inplace=True),
            NoisyLinear(512, 1, sigma_init)
        )
        
        # 優勢流 (動作優勢 A(s,a))
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, sigma_init),
            nn.ReLU(inplace=True),
            NoisyLinear(512, num_actions, sigma_init)
        )
    
    def _get_conv_output(self, shape):
        """計算卷積層輸出的平坦特徵維度"""
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))
    
    def forward(self, x):
        """前向傳播結合值流和優勢流"""
        features = self.conv_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """重置所有噪聲層的噪聲"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class Agent:
    """基於DQN的代理，用於控制Super Mario Bros遊戲"""
    def __init__(self):
        # 基本設置
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 幀堆疊處理
        self.frame_stack = 4
        self.frame_buffer = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(np.zeros((84, 90), dtype=np.float32))
        
        # 跳幀設置
        self.skip_frames = 3
        self.skip_count = 0
        self.last_action = 0
        
        # 性能監控
        self.step_counter = 0
        self.gc_frequency = 100  # 垃圾回收頻率
        
        # 初始化模型
        self.model = self._init_model()
        
    def _init_model(self):
        """初始化和加載模型"""
        model = DuelingCNN(self.frame_stack, 12).to(self.device)
        
        try:
            model.load_state_dict(torch.load('models/rainbow_icm_best.pth', 
                                           map_location=self.device))
            print("模型加載成功")
        except Exception as e:
            print(f"模型加載失敗: {e}。請確保路徑正確。")
        
        model.eval()  # 設置為評估模式
        return model

    @staticmethod
    def preprocess_frame(frame):
        """預處理幀：灰度化、調整大小、歸一化"""
        try:
            # 使用更快的處理方法
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        except Exception as e:
            print(f"幀處理錯誤: {e}")
            return np.zeros((84, 90), dtype=np.float32)

    def act(self, observation):
        """根據當前觀察選擇動作"""
        try:
            self.step_counter += 1
            
            # 跳幀處理 - 重用上一個動作以提高效率
            if self.skip_count > 0:
                self.skip_count -= 1
                return self.last_action
            
            # 預處理當前幀
            processed_frame = self.preprocess_frame(observation)
            self.frame_buffer.append(processed_frame)
            
            # 準備輸入張量 (不進行不必要的類型轉換)
            stacked_frames = np.array(self.frame_buffer)
            
            with torch.no_grad():  # 避免計算梯度
                state_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.argmax(1).item()
            
            # 更新跳幀狀態
            self.last_action = action
            self.skip_count = self.skip_frames
            
            # 定期垃圾回收以避免內存泄漏
            if self.step_counter % self.gc_frequency == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            return action
            
        except Exception as e:
            print(f"動作選擇錯誤: {e}")
            return self.action_space.sample()  # 發生錯誤時隨機選擇動作


def create_test_env(render_mode=None):
    """創建Super Mario Bros環境及包裝器"""
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode=render_mode)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env


def test_agent(episodes=5, render=False):
    """測試代理在多個回合中的表現"""
    render_mode = 'human' if render else None
    env = create_test_env(render_mode)
    agent = Agent()
    rewards = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()  # 現在reset方法返回(obs, info)
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        # 運行回合
        while not (done or truncated):
            # 選擇動作
            action = agent.act(state)
            
            # 執行動作
            next_state, reward, done, truncated, info = env.step(action)  # 現在step方法返回五個值
            
            # 更新狀態和獎勵
            state = next_state
            total_reward += reward
            steps += 1
        
        # 回合完成
        duration = time.time() - start_time
        
        # 存儲獎勵
        rewards.append(total_reward)
        
        # 輸出結果
        print(f"回合 {episode} 完成!")
        print(f"總步數: {steps}")
        print(f"總獎勵: {total_reward}")
        print(f"持續時間: {duration:.2f} 秒")
        print("-" * 50)
    
    # 最終統計
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    print("\n測試結果:")
    print(f"運行回合數: {len(rewards)}")
    print(f"平均獎勵: {avg_reward:.2f}")
    print(f"最佳獎勵: {max(rewards) if rewards else 0:.2f}")
    
    # 關閉環境
    env.close()
    
    return rewards


if __name__ == "__main__":
    # 設置為True可視化遊戲過程，False更快進行測試
    render_gameplay = True
    
    # 測試代理
    print("正在測試代理...")
    test_agent(episodes=1, render=render_gameplay)