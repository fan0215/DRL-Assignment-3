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

# Noisy Linear Layer (Evaluation Version - uses only mean weights)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters (mean values)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features)) # Needed for state_dict loading
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features)) # Needed for state_dict loading

        # Reset parameters to initialize weights similar to training
        self.reset_parameters()

        # Buffers are not strictly needed for eval-only forward pass but are part of the saved state_dict
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

    def reset_parameters(self):
        """Initialize the parameters"""
        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Generate factorized Gaussian noise (Not used in eval forward pass)"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Reset noise (Not used in eval forward pass, but might be called)"""
        # This part is not critical for evaluation but harmless if called
        try:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        except Exception as e:
            # Handle potential device issues if noise reset is called unexpectedly
            # print(f"Warning: Could not reset noise during evaluation - {e}")
            pass # Ignore noise reset issues during evaluation


    def forward(self, x):
        """Forward pass with noise disabled for evaluation"""
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

        # Calculate feature size dynamically
        # Use try-except for environments where input shape might differ slightly
        feature_size = 0
        try:
            with torch.no_grad():
                 # Use the expected input shape (4, 84, 90)
                dummy_input = torch.zeros(1, in_channels, 84, 90)
                feature_size = self.conv_layers(dummy_input).shape[1]
        except Exception as e:
            print(f"Warning: Could not dynamically determine feature size. Using default 3136. Error: {e}")
            # Fallback based on common output size for this architecture with 84x84 input,
            # adjusted slightly for 84x90 -> (84x90 -> 20x21 -> 9x9 -> 7x8)
            # Let's recalculate for 84x90:
            # Conv1: (84-8)/4 + 1 = 19+1=20, (90-8)/4 + 1 = 20.5+1=21 -> 20x21x32
            # Conv2: (20-4)/2 + 1 = 8+1=9, (21-4)/2 + 1 = 8.5+1=9 -> 9x9x64
            # Conv3: (9-3)/1 + 1 = 6+1=7, (9-3)/1 + 1 = 6+1=7 -> 7x7x64
            # feature_size = 7 * 7 * 64 # = 3136 (This seems to be the standard size from DQN papers)
            # Let's re-run the dummy input calculation carefully
            # Input: 1x4x84x90
            # Conv1 (k=8, s=4): H_out = floor((84 - 8) / 4) + 1 = floor(76 / 4) + 1 = 19 + 1 = 20
            #                    W_out = floor((90 - 8) / 4) + 1 = floor(82 / 4) + 1 = floor(20.5) + 1 = 20 + 1 = 21
            # Output shape: 1x32x20x21
            # Conv2 (k=4, s=2): H_out = floor((20 - 4) / 2) + 1 = floor(16 / 2) + 1 = 8 + 1 = 9
            #                    W_out = floor((21 - 4) / 2) + 1 = floor(17 / 2) + 1 = floor(8.5) + 1 = 8 + 1 = 9
            # Output shape: 1x64x9x9
            # Conv3 (k=3, s=1): H_out = floor((9 - 3) / 1) + 1 = floor(6 / 1) + 1 = 6 + 1 = 7
            #                    W_out = floor((9 - 3) / 1) + 1 = floor(6 / 1) + 1 = 6 + 1 = 7
            # Output shape: 1x64x7x7
            # Flattened size: 64 * 7 * 7 = 64 * 49 = 3136
            feature_size = 3136
            print(f"Using calculated feature size: {feature_size}")


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
        try:
            features = self.conv_layers(x)
        except Exception as e:
            print(f"Error during conv_layers forward pass: {e}")
            print(f"Input shape: {x.shape}")
            # Attempt to provide info about layer shapes if possible
            current_shape = x.shape
            for i, layer in enumerate(self.conv_layers):
                 try:
                     x_layer = layer(x if i == 0 else x_layer)
                     print(f"Shape after layer {i} ({type(layer).__name__}): {x_layer.shape}")
                     current_shape = x_layer.shape
                 except Exception as le:
                     print(f"Error at layer {i} ({type(layer).__name__}): {le}")
                     print(f"Input shape to this layer: {current_shape}")
                     raise le # Re-raise the specific error
            raise e # Re-raise original error if loop finishes somehow

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
    """Agent that acts using a loaded Rainbow DQN model."""
    def __init__(self):
        """Initialize the agent."""
        # Parameters
        self.action_space = gym.spaces.Discrete(12) # COMPLEX_MOVEMENT has 12 actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Agent using device: {self.device}")
        self.frame_stack = 4
        self.frame_buffer = deque(maxlen=self.frame_stack)
        # Use skip_frames = 4 consistent with training's SkipFrame wrapper
        self.skip_frames = 4
        self.skip_count = 0
        self.last_action = 0 # Default action (e.g., NOOP or right)

        # Initialize frame buffer with zeros
        self.reset() # Call reset to initialize buffer correctly

        # --- Model Initialization ---
        # Use the same DuelingCNN architecture as in training
        self.model = DuelingCNN(self.frame_stack, self.action_space.n).to(self.device)

        # --- Model Loading ---
        model_loaded = False
        model_paths_to_try = [
            'models/rainbow_icm_best.pth', # Preferred path from training script
            'models/rainbow_icm_final.pth',
            'models/rainbow_icm_model.pth',
            'rainbow_icm_best.pth',        # Current directory fallbacks
            'rainbow_icm_final.pth',
            'rainbow_icm_model.pth'
        ]

        for path in model_paths_to_try:
            if os.path.exists(path):
                try:
                    print(f"Attempting to load model from: {path}")
                    # Load state dict; map_location ensures compatibility if trained on GPU but run on CPU
                    state_dict = torch.load(path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"Successfully loaded model weights from {path}")
                    model_loaded = True
                    break # Exit loop once model is loaded
                except Exception as e:
                    print(f"Failed to load model from {path}: {e}")
            else:
                 print(f"Model path not found: {path}")

        if not model_loaded:
            print("------------------------------------------------------")
            print("WARNING: Failed to load any pre-trained model weights.")
            print("The agent will act randomly or based on initial weights.")
            print("Ensure 'models/rainbow_icm_best.pth' or another model file exists.")
            print("------------------------------------------------------")
            # Optional: Initialize model weights randomly if no load?
            # The DuelingCNN init already does some initialization.

        # Set the model to evaluation mode
        # This disables dropout layers and affects batch normalization layers.
        # Crucially, it signals NoisyLinear layers (if implemented correctly for eval)
        # to use the mean weights and disable noise.
        self.model.eval()

    def preprocess_frame(self, frame):
        """Convert RGB frame to Grayscale, Resize and Normalize.
           Matches the GrayScaleResize wrapper from training.
        Args:
            frame (np.ndarray): Input RGB frame (H, W, C).
        Returns:
            np.ndarray: Processed frame (84, 90), float32, normalized [0, 1].
        """
        if frame is None:
            print("Warning: Received None frame in preprocess_frame.")
            # Return a zero frame matching the expected shape and type
            return np.zeros((84, 90), dtype=np.float32)

        try:
             # Ensure frame is in HWC format (common for Gym)
            if frame.shape[-1] != 3:
                 # Attempt to handle cases where channel might be first or missing
                 if frame.shape[0] == 3: # CHW -> HWC
                     frame = np.transpose(frame, (1, 2, 0))
                 elif len(frame.shape) == 2: # Grayscale already? -> Add channel dim
                     frame = np.expand_dims(frame, axis=-1)
                     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Convert to RGB for consistency
                 else:
                    raise ValueError(f"Unexpected frame shape for preprocessing: {frame.shape}")


            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Resize to 84x90 using INTER_AREA for downscaling
            resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)

            # Normalize to [0.0, 1.0] and convert to float32
            normalized = resized.astype(np.float32) / 255.0

            return normalized

        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            print(f"Input frame shape: {frame.shape}")
            # Return a zero frame as fallback
            return np.zeros((84, 90), dtype=np.float32)
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
            print(f"Input frame shape: {frame.shape}")
             # Return a zero frame as fallback
            return np.zeros((84, 90), dtype=np.float32)

    def _get_stacked_state(self):
        """Stack frames from the buffer."""
        # The deque automatically handles the fixed size. Convert to numpy array.
        return np.array(self.frame_buffer)

    def act(self, observation):
        """Select an action based on the current observation.
        Args:
            observation (np.ndarray): Current environment observation (RGB frame).
        Returns:
            int: The action selected by the agent.
        """
        # Frame Skipping Logic
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action # Repeat the last chosen action

        # Process the frame if it's time to decide a new action
        processed_frame = self.preprocess_frame(observation)

        # Add the processed frame to the buffer
        self.frame_buffer.append(processed_frame)

        # Get the stacked state
        state = self._get_stacked_state() # Shape: (4, 84, 90)

        # Convert state to tensor, add batch dimension, move to device
        # Use try-except for potential issues during tensor conversion
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Shape: (1, 4, 84, 90)
        except Exception as e:
            print(f"Error converting state to tensor: {e}")
            print(f"State shape before conversion: {state.shape}")
            # Fallback to a random action or default action
            self.last_action = self.action_space.sample()
            self.skip_count = self.skip_frames - 1 # Reset skip count
            return self.last_action


        # Select action using the model
        with torch.no_grad(): # Disable gradient calculation for inference
            q_values = self.model(state_tensor)
            # Select the action with the highest Q-value
            action = q_values.max(1)[1].item() # .item() extracts the scalar value

        # Store the chosen action and reset skip count
        self.last_action = action
        self.skip_count = self.skip_frames - 1 # Start skipping for the next frames

        # The agent acts on this frame, then skips `skip_frames - 1` times
        return action

    def reset(self):
        """Reset the agent's internal state (frame buffer, skip count)."""
        self.frame_buffer.clear()
        # Fill the buffer with zero frames initially
        for _ in range(self.frame_stack):
            self.frame_buffer.append(np.zeros((84, 90), dtype=np.float32))
        self.skip_count = 0
        self.last_action = 0 # Reset last action (optional, could keep last)
        gc.collect() # Optional garbage collection
        print("Agent state reset.")

# Example Usage (Optional, for testing the Agent class directly)
# if __name__ == '__main__':
#     # Create a dummy environment matching the expected observation space
#     class DummyEnv:
#         def __init__(self):
#             self.observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
#         def reset(self):
#             return self.observation_space.sample()
#         def step(self, action):
#             obs = self.observation_space.sample()
#             reward = random.random()
#             done = random.random() < 0.01
#             info = {}
#             return obs, reward, done, info

#     env = DummyEnv()
#     agent = Agent()

#     obs = env.reset()
#     agent.reset() # Ensure agent is reset before starting

#     total_reward = 0
#     for step in range(100):
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#         print(f"Step: {step}, Action: {action}, Reward: {reward:.2f}, Done: {done}")
#         total_reward += reward
#         if done:
#             print("Episode finished.")
#             obs = env.reset()
#             agent.reset() # Reset agent when env resets

#     print(f"Total reward over 100 steps: {total_reward}")