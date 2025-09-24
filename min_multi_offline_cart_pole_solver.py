import tensorflow as tf
import numpy as np
from tensorflow import keras
from rich.console import Console
from rich.progress import track
import time
import argparse
import gymnasium as gym

nprng = np.random.default_rng()
console = Console()

class MCDropoutLayer(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    
def nll_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    log_var = y_pred[:, 1]
    var = tf.math.exp(log_var)
    return 0.5*log_var+(1/(2*var))*(y_true-mu)**2

def make_traditional_action_model(state_dim, action_dim):
    flat_input = keras.layers.Input(shape=(state_dim,))
    #flat_norm = keras.layers.BatchNormalization(axis=1)(flat_input)
    
    dense_1 = keras.layers.Dense(
        128,
        #kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(flat_input)
    #dense_1_norm = keras.layers.BatchNormalization()(dense_1)
    #drop_1 = MCDropoutLayer(0.2)(dense_1_norm)

    dense_2 = keras.layers.Dense(
        128,
        #kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(dense_1)
    #dense_2_norm = keras.layers.BatchNormalization()(dense_2)
    #drop_2 = MCDropoutLayer(0.2)(dense_2_norm)

    output = keras.layers.Dense(
        action_dim,
        #kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
    )(dense_2)

    model = keras.Model(inputs=flat_input, outputs=output)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3)
    )
    return model

class CartPoleAgent():
    def __init__(self, flat_dim, action_dim):
        self.Q_model = make_traditional_action_model(
            state_dim=flat_dim,
            action_dim=action_dim
            #n_peaks=self.n_peaks,
        )

        self.Q_prime_1_model = make_traditional_action_model(
            state_dim=flat_dim,
            action_dim=action_dim
            #n_peaks=self.n_peaks,
        )

        self.Q_prime_2_model = make_traditional_action_model(
            state_dim=flat_dim,
            action_dim=action_dim
            #n_peaks=self.n_peaks,
        )

    def get_max_Q_traditional_model_action(self, state):
        Q_predictions = self.Q_model.predict(np.array([state]), verbose=0)

        best_Q_value = np.max(Q_predictions)
        action = np.argmax(Q_predictions)
        return best_Q_value, action
    
    def get_max_Q_traditional_model_actions(self, states):
        Q_predictions = self.Q_model.predict(states, verbose=0)
        best_Q_value = np.max(Q_predictions, axis=1)
        action = np.argmax(Q_predictions, axis=1)
        return best_Q_value, action

    def get_min_Q_prime_traditional_model_value(self, state, action):
        action = np.array([action])
        Q_prime_1_predictions = self.Q_prime_1_model.predict(np.array([state]), verbose=0)
        Q_prime_2_predictions = self.Q_prime_2_model.predict(np.array([state]), verbose=0)
        avg_Q_prime_predictions = np.min([Q_prime_1_predictions, Q_prime_2_predictions], axis=0)
        Q_prime_value = avg_Q_prime_predictions[0, action]
        return Q_prime_value

    def get_min_Q_traditional_prime_model_values(self, states, actions):
        #console.print("Getting Q prime multi states")
        Q_prime_1_predictions = self.Q_prime_1_model.predict(states, verbose=0)
        Q_prime_2_predictions = self.Q_prime_2_model.predict(states, verbose=0)
        avg_Q_prime_predictions = np.min([Q_prime_1_predictions, Q_prime_2_predictions], axis=0)
        Q_prime_values = avg_Q_prime_predictions[np.arange(len(avg_Q_prime_predictions)), actions]
        return Q_prime_values

class ReplayBuffer:
    def __init__(self, capacity, flat_state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.flat_states = np.zeros((capacity, flat_state_dim), dtype=np.float32)
        self.actions=np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_flat_states = np.zeros((capacity,flat_state_dim), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, flat_state, action, reward, done, next_flat_state, priority):
        i = self.ptr
        self.flat_states[i] = flat_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.next_flat_states[i] = next_flat_state
        self.priorities[i] = priority

        # Move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        idxs = nprng.choice(self.size, size=batch_size, replace=False)
        return (
            self.flat_states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_flat_states[idxs],
            self.priorities[idxs],
            idxs,
        )

    def weighted_sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        probs = self.priorities[:self.size]
        probs = probs / np.sum(probs)
        idxs = nprng.choice(self.size, size=batch_size, p=probs, replace=False)
        return (
            self.flat_states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_flat_states[idxs],
            self.priorities[idxs],
            idxs
        )

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities

def main(args):
    console.log('Setting up env and agent')
    env = gym.make("CartPole-v1")
    flat_dim=4
    action_dim=2
    agent = CartPoleAgent(
        flat_dim=flat_dim,
        action_dim=action_dim,
    )
    agent.Q_prime_1_model.set_weights(agent.Q_model.get_weights())

    console.log('Running training of the Q model')
    replay_buffer = ReplayBuffer(capacity = 20000, flat_state_dim=flat_dim, action_dim=action_dim)

    gamma = 0.99
    alpha=0.5
    beta=0.0
    beta_annealing_factor = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_rate = 0.995

    total_episodes = 500
    replay_buffer_warmup_period = 1000
    batch_size=128
    Q_prime_dilation=3
    Q_prime_delay = 200
    # new training loop
    sums_of_rewards = []
    reward_sum_queue = []
    reward_sum_queue_length = 10

    #new style loopwith fixes
    episode_num = 0
    total_steps = 0
    for episode_num in track(range(total_episodes), console=console):
        start_time = time.perf_counter()
        obs, info = env.reset()

        done = False
        rewards=[]
        epsilon = epsilon_end + (epsilon_start - epsilon_end)*(epsilon_decay_rate**episode_num)
        Q_choices=0
        epsilon_choices=0

        while not done:
            total_steps += 1
            epsilon_choice = nprng.random()
            if epsilon_choice < epsilon:
                #Q_value, _ = agent.get_max_Q_traditional_model_action(state=obs)
                #It gets used in sampling weighting
                action = env.action_space.sample() #nprng.integers(0, 1, endpoint=True)
                Q_value = agent.Q_model.predict(np.array([obs]), verbose=0)[0, action]
                epsilon_choices+=1
            else:
                Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
                Q_choices+=1
            # Q_value, action = agent.get_max_Q_model_action(
            #     state=obs,
            # )
            #Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
            flat_array=np.append(obs, action)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            rewards.append(int(reward))

            _, next_action = agent.get_max_Q_traditional_model_action(state=next_obs)

            Q_prime_value = agent.get_min_Q_prime_traditional_model_value(state=next_obs, action=next_action)
            target=reward+gamma*(1.0-done)*Q_prime_value
            #td_error = ((np.abs(Q_value-target)+1e-7)**alpha)[0]
            td_error = float(abs(Q_value-target)[0]+1e-8)

            replay_buffer.add(
                flat_state=obs,
                action=action,
                reward=reward,
                done=int(done),
                next_flat_state=next_obs,
                priority=td_error,
            )

            obs = next_obs

            if total_steps >= replay_buffer_warmup_period:
                flat_states, actions, buffer_rewards, dones, next_flat_states, priorities, idxs = replay_buffer.weighted_sample(batch_size=batch_size)
                #flat_states, actions, buffer_rewards, dones, next_flat_states, priorities, idxs = replay_buffer.sample(batch_size=batch_size)
                #state_action_array = np.append(flat_states, actions, axis=1)
                _, next_actions = agent.get_max_Q_traditional_model_actions(next_flat_states)
                Q_primes = agent.get_min_Q_traditional_prime_model_values(next_flat_states, next_actions)
                target_array = buffer_rewards+(1.0-dones)*gamma*Q_primes
                final_Q_targets=agent.Q_model.predict(flat_states, verbose=0)
                #console.print(final_Q_targets.shape)
                #console.print(final_Q_targets[np.arange(batch_size), actions.astype(np.int32)].shape)
                #console.print(target_array.shape)
                new_td_error = final_Q_targets[np.arange(batch_size), actions.astype(np.int32)] - target_array
                #console.print(new_td_error.shape)
                new_td_error = np.abs(new_td_error).astype(np.float64)
                final_Q_targets[np.arange(batch_size), actions.astype(np.int32)] = target_array

                bias_weights = (1.0/(len(priorities)*(priorities/np.sum(priorities))))**beta
                bias_weights/=np.max(bias_weights)
                beta = 1.0-(beta_annealing_factor**(episode_num))
                # agent.Q_model.train_on_batch(
                #     x=state_action_array,
                #     y=target_array,
                #     #sample_weight=bias_weights
                # )
                agent.Q_model.fit(
                    x=flat_states,
                    y=final_Q_targets,
                    #sample_weight=bias_weights,
                    verbose=0,
                )
                replay_buffer.update_priorities(idxs, new_td_error)
                if total_steps % Q_prime_delay == 0:
                    agent.Q_prime_1_model.set_weights(agent.Q_model.get_weights())
                if ((total_steps+(Q_prime_delay // 2)) % Q_prime_delay) == 0:
                    agent.Q_prime_2_model.set_weights(agent.Q_model.get_weights())
                    
        end_time = time.perf_counter()
        sums_of_rewards.append(np.sum(rewards))
        if len(reward_sum_queue) < reward_sum_queue_length:
            reward_sum_queue.append(np.sum(rewards))
        else:
            reward_sum_queue[episode_num % reward_sum_queue_length] = np.sum(rewards)
        console.log()
        console.log(f'Episode: {episode_num}, Episode Length: {end_time-start_time:.2f}s, Total Steps: {total_steps}')
        console.log(f'# epsilon action: {epsilon_choices}, #Q actions: {Q_choices}, Q/total: {Q_choices/(Q_choices+epsilon_choices):.2%}, epsilon: {epsilon:.2f}')
        console.log(f'Sum of rewards: {np.sum(rewards)}, Running Avg of Sum: {np.mean(sums_of_rewards):.2f}, Avg of {reward_sum_queue_length} Reward Sums: {np.mean(reward_sum_queue):.2f}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training of the trading model")
    parser.add_argument(
        '--load_models',
        action='store_true',
        help='Load models instead of creating new ones'
    )
    parser.add_argument(
        '--episodes',
        default=7500,
        nargs='?',
        help='Episodes to run training for',
        type=int,
    )
    parser.add_argument(
        '--episodes_elapsed',
        default=0,
        nargs='?',
        help='If resuming training, set training parameters accordingly',
        type=int,
    )

    args=parser.parse_args()
    
    main(args)
