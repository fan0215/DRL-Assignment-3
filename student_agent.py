import random
import cv2
import os
import gc
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from collections import deque, namedtuple
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Noisy Linear Layer 
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
        
        # Register buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize the parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Generate factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset the factorized noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Forward pass with noise"""
        # During evaluation (for the agent), only use the mean values
        weight = self.weight_mu
        bias = self.bias_mu
        
        return F.linear(x, weight, bias)


# Dueling CNN architecture
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
        
        # Calculate feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 90)
            feature_size = self.conv_layers(dummy_input).shape[1]
        
        # Value stream (state value V(s))
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma_init)
        )
        
        # Advantage stream (action advantage A(s,a))
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma_init)
        )
    
    def forward(self, x):
        """Forward pass combining value and advantage streams"""
        features = self.conv_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        # Reset noise in value stream
        for module in self.value_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise in advantage stream
        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class Agent(object):
    """Agent that acts using a loaded DQN model."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_stack = 4
        self.frame_buffer = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(np.zeros((84, 90), dtype=np.float32))
        self.skip_frames = 3
        self.skip_count = 0
        self.last_action = 0
        
        self.step_counter = 0
        
        self.model = DuelingCNN(self.frame_stack, 12).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load('models/rainbow_icm_best.pth', map_location=self.device))
            print("Model loaded")
        except:
            print("Failed to load model. Ensuring compatibility with evaluation.")
            try:
                # Try alternate locations
                model_paths = [
                    'rainbow_icm_best.pth',
                    'best_model.pth',
                    'mario_dueling_dqn.pth'
                ]
                
                for path in model_paths:
                    try:
                        self.model.load_state_dict(torch.load(path, map_location=self.device))
                        print(f"Model loaded from {path}")
                        break
                    except:
                        continue
            except:
                print("All model loading attempts failed. Using untrained model.")
        
        self.model.eval()

    def preprocess_frame(self, frame):
        """Convert RGB to grayscale and resize to 84x90"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)
            
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return np.zeros((84, 90), dtype=np.float32)

    def act(self, observation):
        try:
            self.step_counter += 1
            
            if self.skip_count > 0:
                self.skip_count -= 1
                return self.last_action
                
            processed_frame = self.preprocess_frame(observation)
            
            self.frame_buffer.append(processed_frame)
            
            stacked_frames = np.array(self.frame_buffer)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.argmax(1).item()
            
            self.last_action = action
            self.skip_count = self.skip_frames
            
            if self.step_counter % 50 == 0:
                gc.collect()
                
            return action
            
        except Exception as e:
            print(f"Error in act method: {e}")
            return self.action_space.sample()
