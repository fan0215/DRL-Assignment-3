import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque


# Set device to CPU explicitly for submission
device = torch.device("cpu")


class SkipFrame:
    """Wrapper to skip frames for faster processing"""
    def __init__(self, env, skip):
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ProcessFrame:
    """Process raw frames for agent input"""
    def __init__(self):
        self.frame_stack = deque(maxlen=4)
        
    def process(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize
        normalized = resized / 255.0
        
        # Update frame stack
        self.frame_stack.append(normalized)
        
        # If the stack isn't full yet, duplicate the frame
        while len(self.frame_stack) < 4:
            self.frame_stack.append(normalized)
        
        # Stack frames into a 4-channel tensor
        stacked = np.array(self.frame_stack, dtype=np.float32)
        
        return stacked


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Get size of conv output
        self.conv_output_size = self._get_conv_out_size((input_channels, 84, 84))
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 3:  # If the input is a single frame
            x = x.unsqueeze(0)
            
        conv_out = self.conv(x).view(x.size()[0], -1)
        advantage = self.advantage(conv_out)
        value = self.value(conv_out)
        
        # Combine value and advantage for Q values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class Agent:
    def __init__(self):
        # Initialize frame processor
        self.frame_processor = ProcessFrame()
        
        # Initialize DQN model
        self.model = DuelingDQN(input_channels=4, n_actions=12)
        
        # Load pre-trained model
        try:
            self.model.load_state_dict(torch.load('mario_dueling_dqn.pth', map_location=device))
            print("Model loaded successfully!")
        except:
            print("Warning: Could not load model file. Using unt