import gym
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from collections import deque

# 設定參數與常數
class Config:
    ENV_NAME = "SuperMarioBros-v0"
    IMAGE_SIZE = (84, 84)
    FRAME_SKIP = 4
    FRAME_STACK = 4
    DEATH_PENALTY = -100
    ACTION_SPACE = 12
    MODEL_PATH = "best_model.pth"

# 帶有 noisy layer 的線性層（請確保 NoisyLinear 有定義在其他地方或引入）
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

    def reset_noise(self):
        pass  # 若使用真正的 NoisyNet 實作需補上

# Rainbow DQN 主網路，包含 dueling 架構
class DQNNetwork(nn.Module):
    def __init__(self, input_frames, num_actions):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(input_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        dummy_input = torch.zeros(1, input_frames, *Config.IMAGE_SIZE)
        feature_dim = self.extractor(dummy_input).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 512), nn.ReLU(),
            NoisyLinear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 512), nn.ReLU(),
            NoisyLinear(512, num_actions)
        )

    def forward(self, x):
        features = self.extractor(x).view(x.size(0), -1)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# Mario 智能體：負責接收影格並輸出動作
class Agent:
    def __init__(self, model_path=Config.MODEL_PATH):
        self.device = torch.device("cpu")
        self.frame_skip = Config.FRAME_SKIP
        self.action_space = gym.spaces.Discrete(Config.ACTION_SPACE)
        self.frame_buffer = deque(maxlen=Config.FRAME_STACK)
        self.need_init = True
        self.step_counter = 0
        self.cached_action = 0

        self.policy_net = DQNNetwork(Config.FRAME_STACK, self.action_space.n)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

        # 處理影像的流程
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def _prepare_input(self, frame):
        # 預處理輸入畫面
        tensor = self.preprocess(frame) / 255.0
        return tensor

    def select_action(self, frame):
        # 將畫面轉換為 state 並儲存進 frame buffer
        processed = self._prepare_input(frame)

        if self.need_init:
            self.frame_buffer.clear()
            for _ in range(Config.FRAME_STACK):
                self.frame_buffer.append(processed)
            self.need_init = False
            self.step_counter = 1
            return 0

        self.step_counter = (self.step_counter + 1) % self.frame_skip

        if self.step_counter == 1:
            self.frame_buffer.append(processed)
            with torch.no_grad():
                # 將 frame stack 合併為一個輸入 tensor
                stacked = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
                q_values = self.policy_net(stacked)
                self.cached_action = q_values.argmax(dim=1).item()

        return self.cached_action