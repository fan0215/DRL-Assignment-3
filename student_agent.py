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


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters (mean weights and biases)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Sigma parameters are defined but not used in evaluation mode forward pass
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Reset parameters (initialize means and sigmas)
        self.reset_parameters()

        # Buffers for noise (not used in evaluation mode forward pass)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        # No need to call reset_noise() as we don't use epsilon in eval forward

    def reset_parameters(self):
        """Initialize the parameters"""
        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    # _scale_noise and reset_noise are not needed for evaluation logic,
    # but kept for compatibility if the state_dict includes noise buffers/sigmas.
    def _scale_noise(self, size):
         """Generate factorized Gaussian noise"""
         # Check if weight_mu is on CUDA to determine device
         device = self.weight_mu.device if self.weight_mu.is_cuda else 'cpu'
         x = torch.randn(size, device=device)
         return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
         """Reset the factorized noise"""
         epsilon_in = self._scale_noise(self.in_features)
         epsilon_out = self._scale_noise(self.out_features)

         # Outer product
         self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
         self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Forward pass without noise for evaluation"""
        # During evaluation (self.training is False), use only the mean weights
        weight = self.weight_mu
        bias = self.bias_mu
        return F.linear(x, weight, bias)


# Dueling CNN architecture (Exactly as in train.py, uses the NoisyLinear above)
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
        # Use a temporary device context if necessary, especially for CPU-only environments
        temp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 90, device=temp_device)
            # Temporarily move conv_layers to the temp_device if not already there
            original_device = next(self.conv_layers.parameters()).device
            self.conv_layers.to(temp_device)
            feature_size = self.conv_layers(dummy_input).shape[1]
            self.conv_layers.to(original_device) # Move back if needed


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

    # reset_noise is not strictly needed for evaluation logic if forward pass handles it,
    # but can be kept for consistency or potential future use.
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

# --- Agent Definition ---

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts using a loaded Rainbow DQN model."""
    def __init__(self):
        # Parameters matching the training setup
        self.action_space = gym.spaces.Discrete(12) # COMPLEX_MOVEMENT has 12 actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Agent using device: {self.device}")

        self.frame_stack = 4 # Number of frames to stack
        self.frame_buffer = deque(maxlen=self.frame_stack)
        # Initialize buffer with zeros matching the processed frame shape
        for _ in range(self.frame_stack):
            self.frame_buffer.append(np.zeros((84, 90), dtype=np.float32))

        # Frame skipping settings (matching training or template)
        # Using skip_frames = 4 from make_mario_env default in train.py
        # Note: The template had 3, adjust if needed. Using 4 to match train.py's wrapper.
        self.skip_frames = 4
        self.skip_count = 0
        self.last_action = 0 # Store the last chosen action for skipping

        self.step_counter = 0 # For periodic garbage collection

        # Define the model architecture (using classes defined above)
        # Input channels = frame_stack, num_actions = action_space.n
        self.model = DuelingCNN(self.frame_stack, self.action_space.n).to(self.device)

        # Load the trained model weights
        model_path = 'models/rainbow_icm_best.pth' # Target the best model saved during training
        # Also try the final model if the best isn't found
        fallback_model_path = 'models/rainbow_icm_final.pth'
        # Default path if neither is found (or use the standard name from save())
        default_model_path = 'models/rainbow_icm_model.pth'

        loaded = False
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Successfully loaded model from {model_path}")
                loaded = True
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}. Error: {e}")
        
        if not loaded and os.path.exists(fallback_model_path):
             try:
                self.model.load_state_dict(torch.load(fallback_model_path, map_location=self.device))
                print(f"Successfully loaded model from {fallback_model_path}")
                loaded = True
             except Exception as e:
                 print(f"Warning: Failed to load model from {fallback_model_path}. Error: {e}")

        if not loaded and os.path.exists(default_model_path):
             try:
                self.model.load_state_dict(torch.load(default_model_path, map_location=self.device))
                print(f"Successfully loaded model from {default_model_path}")
                loaded = True
             except Exception as e:
                 print(f"Warning: Failed to load model from {default_model_path}. Error: {e}")


        if not loaded:
            print("Error: Failed to load any model. Agent will act randomly.")
            # Optionally, you could raise an error here or handle it differently
            # raise FileNotFoundError("Could not find a suitable model file.")

        # Set the model to evaluation mode (important!)
        # This disables dropout, batchnorm updates, and uses NoisyLinear means
        self.model.eval()

    def preprocess_frame(self, frame):
        """Convert RGB frame to Grayscale, Resize, and Normalize (like GrayScaleResize wrapper)"""
        try:
            # Ensure frame is NumPy array (if coming directly from gym)
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            # Check if the frame is already grayscale (shape H, W) or RGB (H, W, C)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                 # Convert to grayscale
                 gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif len(frame.shape) == 2:
                 # Assume it's already grayscale
                 gray = frame
            else:
                 # Handle unexpected frame shape
                 print(f"Warning: Unexpected frame shape {frame.shape}. Trying to process.")
                 # Attempt grayscale conversion assuming first 3 channels are RGB if C > 3
                 if len(frame.shape) == 3 and frame.shape[2] >= 3:
                     gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
                 else: # Fallback to zeros
                     print("Error: Cannot process frame shape. Returning zeros.")
                     return np.zeros((84, 90), dtype=np.float32)

            # Resize to 84x90 (Height x Width)
            resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            return normalized
        except Exception as e:
            print(f"Error during preprocessing frame: {e}")
            # Return a dummy frame in case of error
            return np.zeros((84, 90), dtype=np.float32)

    def act(self, observation):
        """Selects an action based on the current observation using the loaded model."""
        try:
            self.step_counter += 1

            # Frame Skipping Logic (like SkipFrame wrapper)
            # If skip_count > 0, we are in the middle of skipping frames, repeat the last action
            if self.skip_count > 0:
                self.skip_count -= 1
                return self.last_action

            # --- If not skipping, process the new observation and select a new action ---

            # Preprocess the raw observation (current frame)
            processed_frame = self.preprocess_frame(observation)

            # Add the processed frame to the buffer (FrameStack logic)
            self.frame_buffer.append(processed_frame)

            # Create the state by stacking frames from the buffer
            # The buffer automatically handles the correct number of frames
            stacked_frames = np.array(self.frame_buffer) # Shape: (frame_stack, H, W)

            # Ensure the stacked_frames has the correct dimensions (e.g., if buffer wasn't full initially)
            if stacked_frames.shape[0] < self.frame_stack:
                # This case should ideally not happen after initialization, but handle defensively
                print("Warning: Frame buffer not full during action selection.")
                # Pad with zeros if necessary (less ideal)
                padding = np.zeros((self.frame_stack - stacked_frames.shape[0], 84, 90), dtype=np.float32)
                stacked_frames = np.concatenate((padding, stacked_frames), axis=0)


            # Convert state to PyTorch tensor, add batch dimension, and move to device
            # Input shape for CNN: (batch_size, channels, height, width) -> (1, frame_stack, 84, 90)
            state_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)

            # Get Q-values from the model (inference)
            with torch.no_grad(): # Disable gradient calculation for inference
                q_values = self.model(state_tensor)

            # Select the action with the highest Q-value (greedy policy for evaluation)
            action = q_values.argmax(1).item() # .item() converts tensor to Python int

            # Store the chosen action and reset the skip counter for the *next* steps
            self.last_action = action
            # We will skip the next `skip_frames - 1` frames. Setting counter to skip_frames means
            # the current action is executed, and then skip_count is decremented.
            # The template had `self.skip_count = self.skip_frames`. Let's stick to that.
            # It means we execute action, then skip 3 times (if skip_frames=4).
            self.skip_count = self.skip_frames -1 # Start countdown for next frames


            # Periodic garbage collection
            if self.step_counter % 100 == 0: # Collect garbage every 100 steps
                gc.collect()

            return action

        except Exception as e:
            print(f"Error in Agent.act method: {e}")
            # Fallback to a random action in case of any unexpected error
            return self.action_space.sample()