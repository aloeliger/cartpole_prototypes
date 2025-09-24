import tensorflow as tf
import numpy as np
from tensorflow import keras
from rich.console import Console
from rich.progress import track
from gymnasium_environment import TradingEnv
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

def make_action_model(flat_shape):
    flat_input = keras.layers.Input(shape=flat_shape)
    flat_norm = keras.layers.LayerNormalization(axis=1)(flat_input)
    
    dense_1 = keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(flat_norm)
    dense_1_norm = keras.layers.LayerNormalization()(dense_1)
    drop_1 = MCDropoutLayer(0.2)(dense_1_norm)

    dense_2 = keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(drop_1)
    dense_2_norm = keras.layers.LayerNormalization()(dense_2)
    drop_2 = MCDropoutLayer(0.2)(dense_2_norm)

    mu = keras.layers.Dense(
        1,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
    )(drop_2)

    log_var = keras.layers.Dense(
        1,
        #activation='softplus',
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
    )(drop_2)
    output = keras.layers.Concatenate()([mu, log_var])

    model = keras.Model(inputs=flat_input, outputs=output)

def make_traditional_action_model(flat_shape):
    flat_input = keras.layers.Input(shape=flat_shape)
    flat_norm = keras.layers.LayerNormalization(axis=1)(flat_input)
    
    dense_1 = keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(flat_norm)
    dense_1_norm = keras.layers.LayerNormalization()(dense_1)
    drop_1 = MCDropoutLayer(0.2)(dense_1_norm)

    dense_2 = keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
        activation='relu',
    )(drop_1)
    dense_2_norm = keras.layers.LayerNormalization()(dense_2)
    drop_2 = MCDropoutLayer(0.2)(dense_2_norm)

    output = keras.layers.Dense(
        1,
        kernel_regularizer=keras.regularizers.L1L2(0.001, 0.1),
    )(drop_2)

    model = keras.Model(inputs=flat_input, outputs=output)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam()
    )
    return model

class CartPoleAgent():
    def __init__(self, flat_dim, action_dim):
        # self.Q_model = make_action_model(
        #     flat_shape=flat_shape,
        #     #n_peaks=self.n_peaks,
        # )

        # self.Q_1_model = make_action_model(
        #     flat_shape=flat_shape,
        #     #n_peaks=self.n_peaks,
        # )

        # self.Q_2_model = make_action_model(
        #     flat_shape=flat_shape,
        #     #n_peaks=self.n_peaks,
        # )

        self.Q_model = make_traditional_action_model(
            flat_shape=(flat_dim+action_dim,),
            #n_peaks=self.n_peaks,
        )

        self.Q_1_model = make_traditional_action_model(
            flat_shape=(flat_dim+action_dim,),
            #n_peaks=self.n_peaks,
        )

        self.Q_2_model = make_traditional_action_model(
            flat_shape=(flat_dim+action_dim,),
            #n_peaks=self.n_peaks,
        )

    def get_max_Q_traditional_model_action(self, state):
        left_array = np.array([np.append(state, 0)])
        right_array = np.array([np.append(state, 0)])

        left_predictions = self.Q_model.predict(left_array, verbose=0)
        right_predictions = self.Q_model.predict(right_array, verbose=0)

        Q_values = [left_predictions[0], right_predictions[0]]
        best_Q_value = np.max(Q_values)
        action = np.argmax(Q_values)
        return best_Q_value, action
    
    def get_max_Q_traditional_model_actions(self, states):
        left_array = np.append(
            states,
            np.zeros(shape=(len(states),1)),
            axis=1
        )
        right_array = np.append(
            states,
            np.ones(shape=(len(states), 1)),
            axis=1
        )
        left_predictions = self.Q_model.predict(left_array, verbose=0)
        right_predictions = self.Q_model.predict(right_array, verbose=0)

        Q_values = np.array([left_predictions, right_predictions])
        best_Q_value = np.max(Q_values, axis=0)
        action = np.argmax(Q_values, axis=0)
        return best_Q_value, action

    def get_min_Q_prime_traditional_model_value(self, state, action):
        action = np.array([action])
        state_action_array = np.array([np.append(
            np.array(state),
            action,
            axis=0
        )])
        Q_prime_1_predictions = self.Q_1_model.predict(state_action_array, verbose=0)
        Q_prime_2_predictions = self.Q_2_model.predict(state_action_array, verbose=0)
        Q_prime=np.min(
            np.array([Q_prime_1_predictions, Q_prime_2_predictions]),
            axis=0
        )
        return Q_prime

    def get_min_Q_traditional_prime_model_values(self, states, actions):
        state_action_array = np.append(
            states,
            actions,
            axis=1
        )
        Q_prime_1_predictions = self.Q_1_model.predict(state_action_array, verbose=0)
        Q_prime_2_predictions = self.Q_2_model.predict(state_action_array, verbose=0)
        Q_prime=np.min(
            np.array([Q_prime_1_predictions, Q_prime_2_predictions]),
            axis=0
        )
        return Q_prime

    def sample_model(self, model_predictions):
        mu = np.mean(model_predictions[:, 0])
        std_stat = np.std(model_predictions[:, 0])
        log_var = model_predictions[:, 1]
        std_syst = np.mean(np.sqrt(np.exp(log_var)))
        std = np.sqrt(std_stat**2+std_syst**2)

        sample = nprng.normal(loc=mu, scale=std)
        return sample

    def model_samples(self, predictions):
        mu = np.mean(predictions[:, :, 0], axis=0) #shape (examples,)
        std_stat = np.std(predictions[:, :, 0], axis=0) #shape (examples,)
        log_var = predictions[:, :, 1] #shape (repeats, examples,)
        std_syst = np.mean(np.sqrt(np.exp(log_var)), axis=0) #shape(examples,)
        std = np.sqrt(std_stat**2+std_syst**2)
        sample = nprng.normal(loc=mu, scale=std)
        return sample
    
    def get_max_Q_model_action(self, state):
        left_array = np.array([np.append(state, 0)])
        right_array = np.array([np.append(state, 0)])

        left_inputs = np.tile(left_array, (50,1))
        right_inputs = np.tile(right_array, (50,1))
        
        left_predictions = self.Q_model.predict(left_inputs, verbose=0)
        right_predictions = self.Q_model.predict(right_inputs, verbose=0)

        left_sample = self.sample_model(left_predictions)
        right_sample = self.sample_model(right_predictions)
        
        Q_values = np.array([left_sample, right_sample])
        best_Q_value = np.max(Q_values)
        action = np.argmax(Q_values)
        
        return best_Q_value, action

    def get_max_Q_model_actions(self, states):
        left_array = np.append(
            states,
            np.zeros(shape=(len(states),1)),
            axis=1
        ) #shape (examples, state vars)
        right_array = np.append(
            states,
            np.ones(shape=(len(states),1)),
            axis-1
        ) #shape (examples, state vars)

        left_predictions = np.array([
            self.Q_model.predict(left_array, verbose=0)
            for _ in range(50)
        ]) #shape (reps, examples, mu or log var)
        right_predictions = np.array([
            self.Q_model.predict(right_array, verbose=0)
            for _ in range(50)
        ])

        left_samples = self.model_samples(left_predictions) #shape = (examples,)
        right_samples = self.model_samples(right_predictions) #shape=(examples,)

        Q_values = np.array([left_samples, right_samples]) #shape (2, examples)
        best_Q_value = np.max(Q_values, axis=0) #shape = (examples,)
        action = np.argmax(Q_values,axis=0) #shape = (examples,)
        return best_Q_value, action
    
    def get_min_Q_prime_model_value(self, state, action):
        action = np.array([action])
        state_action_array = np.array([
            np.append(
                np.array(state),
                action,
                axis=0
            )
        ])
        inputs = np.tile(state_action_array, (50,1))
        Q_prime_1_predictions = self.Q_1_model.predict(inputs, verbose=0)
        Q_prime_2_predictions = self.Q_2_model.predict(inputs, verbose=0)

        Q_prime_1_sample = self.sample_model(Q_prime_1_predictions)
        Q_prime_2_sample = self.sample_model(Q_prime_2_predictions)

        Q_prime = np.min(
            np.array([Q_prime_1_sample, Q_prime_2_sample]),
            axis=0,
        )
        return Q_prime
    
    def get_min_Q_prime_model_values(self, states, actions):
        state_action_array = np.append(
            states,
            actions,
            axis=1
        )
        Q_prime_1_predictions = np.array([
            self.Q_1_model.predict(state_action_array, verbose=0)
            for _ in range(50)
        ])
        Q_prime_2_predictions = np.array([
            self.Q_2_model.predict(state_action_array, verbose=0)
            for _ in range(50)
        ]) #shape(reps, examples, mu or log var)

        Q_prime_1_samples = self.model_samples(Q_prime_1_predictions) #shape (examples,)
        Q_prime_2_samples = self.model_samples(Q_prime_2_predictions)

        Q_prime = np.min(
            np.array([Q_prime_1_samples, Q_prime_2_samples]),
            axis=0
        )        
        return Q_prime

class ReplayBuffer:
    def __init__(self, capacity, flat_state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.flat_states = np.zeros((capacity, flat_state_dim), dtype=np.float32)
        self.actions=np.zeros((capacity, action_dim), dtype=np.float32)
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
        )

def main(args):
    console.log('Setting up env and agent')
    env = gym.make("CartPole-v1")
    flat_dim=4
    action_dim=1
    agent = CartPoleAgent(
        flat_dim=flat_dim,
        action_dim=action_dim,
    )

    console.log('Running training of the Q model')
    replay_buffer = ReplayBuffer(capacity = 20000, flat_state_dim=flat_dim, action_dim=action_dim)

    gamma = 0.99
    alpha=0.5
    beta=0.0
    beta_annealing_factor = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_rate = 0.995

    total_episodes = 7500
    replay_buffer_warmup_period = 200
    batch_size=128
    Q_prime_dilation=3
    Q_prime_delay = 200
    # new training loop
    sums_of_rewards = []
    reward_sum_queue = []
    reward_sum_queue_length = 10
    # for episode_num in track(range(0, total_episodes), console=console):
    #     obs, info = env.reset()
    #     #console.print(obs)
    #     done = False
    #     rewards=[]
    #     Qs = []
    #     Q_primes=[]
    #     targets=[]

    #     epsilon = epsilon_end + (epsilon_start - epsilon_end)*(epsilon_decay_rate**episode_num)
    #     while not done:
    #         epsilon_choice = nprng.random()
    #         if epsilon_choice < epsilon:
    #             Q_value, _ = agent.get_max_Q_traditional_model_action(state=obs)
    #             #It gets used in sampling weighting
    #             action = nprng.integers(0, 1, endpoint=True)
    #         else:
    #             Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
    #         # Q_value, action = agent.get_max_Q_model_action(
    #         #     state=obs,
    #         # )
    #         #Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
    #         Qs.append(Q_value)
    #         flat_array=np.append(obs, action)

    #         next_obs, reward, terminated, truncated, next_info = env.step(action)
    #         done = terminated or truncated
    #         rewards.append(int(reward))

    #         # Q_prime_value, _ = agent.get_min_Q_prime_model_value(
    #         #     state=next_obs
    #         # )
    #         Q_prime_value, _ = agent.get_min_Q_prime_traditional_model_value(
    #             state=next_obs,
    #         )
    #         Q_primes.append(Q_prime_value)

    #         target=reward+gamma*(1.0-done)*Q_prime_value
    #         targets.append(target)

    #         weight=abs(target-Q_value)**alpha

    #         replay_buffer.add(
    #             flat_state=flat_array,
    #             reward=reward,
    #             target=target,
    #             weight=weight,
    #         )
    #     end_time = time.perf_counter()
    #     sums_of_rewards.append(np.sum(rewards))
    #     if len(reward_sum_queue) < reward_sum_queue_length:
    #         reward_sum_queue.append(np.sum(rewards))
    #     else:
    #         reward_sum_queue[episode_num % reward_sum_queue_length] = np.sum(rewards)
    #     console.log()
    #     console.log(f'Episode: {episode_num}, Mean Q: {np.mean(replay_buffer.targets[:replay_buffer.size]):.2f}, Std Q: {np.std(replay_buffer.targets[:replay_buffer.size]):.2f}, Episode Length: {end_time-start_time:.2f}s')
    #     console.log(f'\tSum of rewards: {np.sum(rewards)}, Running Avg of Sum: {np.mean(sums_of_rewards):.2f}, Avg of {reward_sum_queue_length} Reward Sums: {np.mean(reward_sum_queue):.2f}')
    #     console.log(f'\tSum of episode Qs: {np.sum(Qs):.2f}, Sum of episode Q\'s: {np.sum(Q_primes):.2f}')
    #     console.log(f'\tMean of episode Qs: {np.mean(Qs):.2f}, Std: {np.std(Qs):.2f}, Q\'s: {np.mean(Q_primes):.2f} & {np.std(Q_primes):.2f}')
    #     console.log(f'\tSum of episode targets: {np.sum(targets):.2f}, Mean: {np.mean(targets):.2f}, Std: {np.std(targets):.2f}')

    #     state_action_array, _, target_array, priority_weights_array = replay_buffer.weighted_sample(batch_size=batch_size)
    #     priority_probs_array = priority_weights_array/np.sum(priority_weights_array)
    #     bias_weight_correction = (1.0/(batch_size*priority_probs_array))**beta
    #     #let's update beta
    #     beta = 1.0-(beta_annealing_factor**episode_num)
    #     final_weights = bias_weight_correction*priority_weights_array

    #     agent.Q_model.train_on_batch(
    #         x=state_action_array,
    #         y=target_array,
    #         sample_weight = final_weights
    #     )
        
    #     if episode_num % Q_prime_dilation == 0:
    #         agent.Q_1_model.train_on_batch(
    #             x=state_action_array,
    #             y=target_array,
    #             sample_weight = final_weights
    #         )
            
    #         agent.Q_2_model.train_on_batch(
    #             x=state_action_array,
    #             y=target_array,
    #             sample_weight = final_weights
    #         )

    #Old style loop
    # episode_num = 1
    # while episode_num <= total_episodes:
    #     start_time = time.perf_counter()
    #     obs, info = env.reset()
    #     #console.print(obs)
    #     done = False
    #     rewards=[]
    #     epsilon = epsilon_end + (epsilon_start - epsilon_end)*(epsilon_decay_rate**episode_num)
    #     while not done:
    #         epsilon_choice = nprng.random()
    #         if epsilon_choice < epsilon:
    #             Q_value, _ = agent.get_max_Q_traditional_model_action(state=obs)
    #             #It gets used in sampling weighting
    #             action = nprng.integers(0, 1, endpoint=True)
    #         else:
    #             Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
    #         # Q_value, action = agent.get_max_Q_model_action(
    #         #     state=obs,
    #         # )
    #         #Q_value, action = agent.get_max_Q_traditional_model_action(state=obs)
    #         flat_array=np.append(obs, action)

    #         next_obs, reward, terminated, truncated, next_info = env.step(action)
    #         done = terminated or truncated
    #         rewards.append(int(reward))

    #         Q_prime_value, _ = agent.get_min_Q_prime_traditional_model_value(state=next_obs)
    #         target=reward+gamma*(1.0-done)*Q_prime_value
    #         td_error = (np.abs(Q_value-target)+1e-7)**alpha

    #         replay_buffer.add(
    #             flat_state=obs,
    #             action=action,
    #             reward=reward,
    #             done=int(done),
    #             next_flat_state=next_obs,
    #             priority=td_error,
    #         )
    #     end_time = time.perf_counter()
    #     sums_of_rewards.append(np.sum(rewards))
    #     if len(reward_sum_queue) < reward_sum_queue_length:
    #         reward_sum_queue.append(np.sum(rewards))
    #     else:
    #         reward_sum_queue[episode_num % reward_sum_queue_length] = np.sum(rewards)
    #     console.log()
    #     console.log(f'Episode: {episode_num}, Episode Length: {end_time-start_time:.2f}s')
    #     console.log(f'\tSum of rewards: {np.sum(rewards)}, Running Avg of Sum: {np.mean(sums_of_rewards):.2f}, Avg of {reward_sum_queue_length} Reward Sums: {np.mean(reward_sum_queue):.2f}')

    #     if episode_num % Q_model_training_period == 0:
    #         epsilon = epsilon_end + (epsilon_start - epsilon_end)*(epsilon_decay_rate**(episode_num//Q_model_training_period))
    #         console.log('Model training period!')
    #         console.log('Generating targets')
    #         flat_states, actions, rewards, dones, next_flat_states, priorities = replay_buffer.weighted_sample(batch_size=replay_buffer.size)
    #         state_action_array = np.append(flat_states, actions, axis=1)
    #         Q_primes, _ = agent.get_min_Q_traditional_prime_model_values(next_flat_states)
    #         target_array = rewards+(1.0-dones)*gamma*Q_primes

    #         bias_weights = (1.0/(len(priorities)*(priorities/np.sum(priorities))))**beta
    #         bias_weights/=np.max(bias_weights)
    #         beta = 1.0-(beta_annealing_factor**episode_num)
    #         console.log('Fitting')

    #         agent.Q_model.fit(
    #             x=state_action_array,
    #             y=target_array,
    #             epochs=3,
    #             validation_split=0.4,
    #             sample_weight=bias_weights
    #         )

    #         if episode_num %(Q_model_training_period * Q_prime_dilation) == 0:
    #             console.log('Fitting Q Primes')
    #             #Update the offline models
    #             agent.Q_1_model.set_weights(agent.Q_model.get_weights())
    #             agent.Q_2_model.set_weights(agent.Q_model.get_weights())
    #             agent.Q_1_model.fit(
    #                 x=state_action_array,
    #                 y=target_array,
    #                 epochs=3,
    #                 validation_split=0.4,
    #                 sample_weight=bias_weights
    #             )
    #             agent.Q_2_model.fit(
    #                 x=state_action_array,
    #                 y=target_array,
    #                 epochs=3,
    #                 validation_split=0.4,
    #                 sample_weight=bias_weights
    #             )
    #     episode_num+=1

    #new style loopwith fixes
    episode_num = 1
    total_steps = 0
    for episode_num in track(range(total_episodes), console=console):
        start_time = time.perf_counter()
        obs, info = env.reset()
        #console.print(obs)
        done = False
        rewards=[]
        epsilon = epsilon_end + (epsilon_start - epsilon_end)*(epsilon_decay_rate**episode_num)
        Q_choices=0
        epsilon_choices=0
        while not done:
            total_steps += 1
            epsilon_choice = nprng.random()
            if epsilon_choice < epsilon:
                Q_value, _ = agent.get_max_Q_traditional_model_action(state=obs)
                #It gets used in sampling weighting
                action = nprng.integers(0, 1, endpoint=True)
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
            td_error = (np.abs(Q_value-target)+1e-7)**alpha

            replay_buffer.add(
                flat_state=obs,
                action=action,
                reward=reward,
                done=int(done),
                next_flat_state=next_obs,
                priority=td_error,
            )

            if total_steps >= replay_buffer_warmup_period:
                flat_states, actions, buffer_rewards, dones, next_flat_states, priorities = replay_buffer.weighted_sample(batch_size=batch_size)
                flat_states, actions, buffer_rewards, dones, next_flat_states, priorities = replay_buffer.sample(batch_size=batch_size)
                state_action_array = np.append(flat_states, actions, axis=1)
                _, next_actions = agent.get_max_Q_traditional_model_actions(next_flat_states)
                Q_primes = agent.get_min_Q_traditional_prime_model_values(next_flat_states, next_actions)
                target_array = buffer_rewards+(1.0-dones)*gamma*Q_primes

                bias_weights = (1.0/(len(priorities)*(priorities/np.sum(priorities))))**beta
                bias_weights/=np.max(bias_weights)
                # agent.Q_model.train_on_batch(
                #     x=state_action_array,
                #     y=target_array,
                #     #sample_weight=bias_weights
                # )
                agent.Q_model.fit(
                    x=state_action_array,
                    y=target_array,
                    #sample_weight=bias_weights
                    verbose=0,
                )
                if total_steps % Q_prime_delay == 0:
                    agent.Q_1_model.set_weights(agent.Q_model.get_weights())
                    agent.Q_2_model.set_weights(agent.Q_model.get_weights())
                    
                    # agent.Q_1_model.train_on_batch(
                    #     x=state_action_array,
                    #     y=target_array,
                    #     #sample_weight=bias_weights
                    # )
                    
                    # agent.Q_2_model.train_on_batch(
                    #     x=state_action_array,
                    #     y=target_array,
                    #     #sample_weight=bias_weights
                    # )
        end_time = time.perf_counter()
        sums_of_rewards.append(np.sum(rewards))
        if len(reward_sum_queue) < reward_sum_queue_length:
            reward_sum_queue.append(np.sum(rewards))
        else:
            reward_sum_queue[episode_num % reward_sum_queue_length] = np.sum(rewards)
        console.log()
        console.log(f'Episode: {episode_num}, Episode Length: {end_time-start_time:.2f}s, Total Steps: {total_steps}')
        console.log(f'\t# epsilon action: {epsilon_choices}, #Q actions: {Q_choices}, Q/total: {Q_choices/(Q_choices+epsilon_choices):.2%}, epsilon: {epsilon:.2f}')
        console.log(f'\tSum of rewards: {np.sum(rewards)}, Running Avg of Sum: {np.mean(sums_of_rewards):.2f}, Avg of {reward_sum_queue_length} Reward Sums: {np.mean(reward_sum_queue):.2f}')
        
        beta = 1.0-(beta_annealing_factor**episode_num)
        console.log('Next Episode!')

        
if __name__ =='__main__':
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

