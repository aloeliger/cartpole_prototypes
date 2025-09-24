import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- Q-network builder ----
def build_q_network(state_shape, n_actions):
    model = keras.Sequential([
        layers.InputLayer(state_shape),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_actions, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model

# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---- Training loop ----
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape
n_actions = env.action_space.n

q_net = build_q_network(state_shape, n_actions)
target_net = build_q_network(state_shape, n_actions)
target_net.set_weights(q_net.get_weights())

buffer = ReplayBuffer()
gamma = 0.99
epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
batch_size = 64
target_update_freq = 200  # steps
train_start = 1000        # warmup before training

total_steps = 0
episodes = 500

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        total_steps += 1

        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_net.predict(state[np.newaxis], verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add((state, action, reward, next_state, done))
        state = next_state
        ep_reward += reward

        # Training step
        if len(buffer) >= train_start:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # --- Double DQN update ---
            # Main net chooses the best action
            next_q_values = q_net.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q_values, axis=1)

            # Target net evaluates that action
            target_q_values = target_net.predict(next_states, verbose=0)
            target_values = rewards + (1 - dones) * gamma * target_q_values[np.arange(batch_size), next_actions]

            # Compute targets for current states
            q_values = q_net.predict(states, verbose=0)
            q_values[np.arange(batch_size), actions] = target_values

            # Gradient step
            q_net.fit(states, q_values, verbose=0)

        # Periodically update target network
        if total_steps % target_update_freq == 0:
            target_net.set_weights(q_net.get_weights())

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {ep}, Reward: {ep_reward}, Epsilon: {epsilon:.3f}")
