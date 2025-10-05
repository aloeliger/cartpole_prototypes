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

def make_quantile_loss(n_quantiles):
    quantiles = tf.range(
        start = 1.0/n_quantiles,
        limit = 1.0,
        delta = 1.0/n_quantiles,
    )
    
    def quantile_pinball_loss(y_true, y_pred):
        #y_true shape = (examples, heads, actions)
        #y_pred = (examples, heads, actions, quantiles)
        loss_value = tf.where(
            y_true[..., tf.newaxis] >= y_pred, #shape (examples, heads, actions, quantiles)
            quantiles[tf.newaxis, tf.newaxis, tf.newaxis, :]*(y_true[..., tf.newaxis]-y_pred), #shape should be (examples, heads, actions, quantiles)
            (1.0-quantiles[tf.newaxis, tf.newaxis, tf.newaxis, :])*(y_pred-y_true[..., tf.newaxis]) #shape should be (examples, heads, actions, quantiles)
        )
        loss_value = tf.reduce_mean(loss_value, axis=[1,2,3])
        return loss_value
    return quantile_pinball_loss

def nll_loss(y_true, y_pred):
    #y_true shape = (examples, heads, actions)
    #y_pred = (examples, heads, actions, mu or log_var)
    mu = y_pred[:, :, :, 0] #shape = (examples, heads, actions)
    log_var = y_pred[:, :, :, 1]+1e-8
    var = tf.math.exp(log_var)
    nll = 0.5*log_var+(1/(2*var))*(y_true-mu)**2 #shape=(examples, heads, actions)
    nll = tf.reduce_sum(nll, axis=(1,2))
    return nll

def make_bootstrap_action_model(state_dim, action_dim, n_heads, n_quantiles, prior_strength=1.0):
    flat_input = keras.layers.Input(shape=(state_dim,))

    n_quantile_heads = n_quantiles-1
    
    dense_1 = keras.layers.Dense(
        128,
        activation='relu',
    )(flat_input)
    dense_2 = keras.layers.Dense(
        128,
        activation='relu',
    )(dense_1)

    quantile_head = keras.layers.Dense(
        action_dim*n_heads*n_quantile_heads,
        activation='linear'
    )(dense_2)
    quantile_prior = keras.layers.Dense(
        action_dim*n_heads*n_quantile_heads,
        kernel_initializer='orthogonal',
        bias_initializer='random_normal',
        trainable=False,
    )(dense_2)
    quantile_head = keras.layers.Lambda(
        lambda z: z[0]+prior_strength*z[1]
    )([quantile_head, quantile_prior])
    quantile_reshape = keras.layers.Reshape(
        target_shape=(n_heads, action_dim, n_quantile_heads)
    )(quantile_head)
    #shape = (examples, head, action, quantiles)

    model = keras.Model(inputs=flat_input, outputs=quantile_reshape)
    quantile_loss = make_quantile_loss(n_quantiles)
    model.compile(
        loss=quantile_loss,
        optimizer=keras.optimizers.Adam(learning_rate=1e-3)
    )
    return model

class CartPoleAgent():
    def __init__(self, flat_dim, action_dim, n_heads=10, n_quantiles=10):
        self.Q_model = make_bootstrap_action_model(
            state_dim=flat_dim,
            action_dim=action_dim,
            n_quantiles=n_quantiles,
            n_heads=n_heads,
        )

        self.Q_prime_1_model = make_bootstrap_action_model(
            state_dim=flat_dim,
            action_dim=action_dim,
            n_quantiles=n_quantiles,
            n_heads=n_heads
        )
        self.Q_prime_2_model=make_bootstrap_action_model(
            state_dim=flat_dim,
            action_dim=action_dim,
            n_quantiles=n_quantiles,
            n_heads=n_heads,
        )

    def get_max_Q_model_action(self, state, head):
        Q_predictions = self.Q_model.predict(np.array([state]), verbose=0)
        #shape = (examples, heads, actions, quantiles)

        #Update: I think this is incorrect. The best Q value is way underestimated here
        #The _action_ needs to be selected with respect to the best worst outcome
        #but the Q value is still the average for that action
        
        #The best Q_value is now the one with the best low quantile
        #nominally, if we know the number of quantiles we have
        #We can calculate exact sets of quantiles instead of taking the last
        #but this will serve for now

        # best_Q_value=np.max(Q_predictions[0, head, :, 0])
        #shape: flat single value
        # action = np.argmax(Q_predictions[0, head, :, 0])
        #shape: flat single value

        #I think this is ultimately more correct
        action = np.argmax(Q_predictions[0, head, :, 0])
        #shape flat single value
        best_Q_value = np.mean(Q_predictions[0, head, action, :])
        #shape flat single value
        
        return best_Q_value, action

    def get_max_Q_model_actions(self, states, heads):
        Q_predictions = self.Q_model.predict(states, verbose=0)
        #shape = (examples, heads, actions, quantiles)

        #Update: I think this is incorrect. The best Q value is way underestimated here
        #The _action_ needs to be selected with respect to the best worst outcome
        #but the Q value is still the average for that action
        
        #The best Q_value is now the one with the best low quantile
        #nominally, if we know the number of quantiles we have
        #We can calculate exact sets of quantiles instead of taking the last
        #but this will serve for now
        
        # best_Q_values = np.max(
        #     Q_predictions[np.arange(len(Q_predictions)), heads, :, 0],
        #     axis=1
        # )
        #shape: (examples,)
        # action = np.argmax(
        #     Q_predictions[np.arange(len(Q_predictions)), heads, :, 0],
        #     axis=1
        # )

        #I think this is ultimately more correct
        action = np.argmax(
            Q_predictions[np.arange(len(Q_predictions)), heads, :, 0],
            axis=-1
        )
        #shape = (examples,)
        best_Q_values = np.mean(
            Q_predictions[np.arange(len(Q_predictions)), heads, action, 0],
            axis=-1
        )
        #shape = (examples,)
        
        return best_Q_values, action

    def get_Q_prime_model_value(self, state, action, head):
        action = np.array([action])
        Q_prime_1_predictions = self.Q_prime_1_model.predict(
            np.array([state]),
            verbose=0,
        )
        #shape (examples, heads, actions, quantiles)
        #this value is trickier
        #The way the loss works, we need a single central value to
        #feed into the loss
        #So the idea here would be to simply take an average over the quantiles
        #This is not exact, and misses the effect from extreme tails,
        #but should be a suitable approximation of a central value
        Q_prime_1_predictions = np.mean(
            Q_prime_1_predictions,
            axis=-1,
        )
        #shape = (examples, heads, actions)
    
        Q_prime_2_predictions = self.Q_prime_2_model.predict(
            np.array([state]),
            verbose=0,
        )
        Q_prime_2_predictions = np.mean(
            Q_prime_2_predictions,
            axis=-1,
        )
        #shape = (examples, heads, actions)
        
        Q_prime_value = np.min([Q_prime_1_predictions, Q_prime_2_predictions], axis=0)[0, head, action]
        return Q_prime_value

    def get_Q_prime_model_values(self, states, actions, heads):
        Q_prime_1_predictions=self.Q_prime_1_model.predict(states, verbose=0)
        #shape (examples, heads, actions, quantiles)
        Q_prime_1_predictions = np.mean(
            Q_prime_1_predictions,
            axis=-1,
        )
        #shape (examples, heads, actions)
        
        Q_prime_2_predictions=self.Q_prime_2_model.predict(states, verbose=0)
        Q_prime_2_predictions = np.mean(
            Q_prime_2_predictions,
            axis=-1,
        )
        #shape ( examples, heads, actions)
        
        Q_prime_values = np.min([Q_prime_1_predictions, Q_prime_2_predictions], axis=0)[np.arange(len(Q_prime_1_predictions)), heads, actions]
        return Q_prime_values

    def ema_update(self, tau_1, tau_2):
        source_weights = self.Q_model.get_weights()
        Q_1_weights = self.Q_prime_1_model.get_weights()
        Q_2_weights = self.Q_prime_2_model.get_weights()

        new_1_weights = [
            tau_1*sw + (1.0-tau_1)*tw
            for sw, tw in zip(source_weights, Q_1_weights)
        ]
        new_2_weights = [
            tau_2*sw+(1.0-tau_2)*tw
            for sw, tw in zip(source_weights, Q_2_weights)
        ]

        self.Q_prime_1_model.set_weights(new_1_weights)
        self.Q_prime_2_model.set_weights(new_2_weights)

class ReplayBuffer:
    def __init__(self, capacity, flat_state_dim, action_dim, n_heads):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.flat_states = np.zeros((capacity, flat_state_dim), dtype=np.float32)
        self.actions=np.zeros((capacity,), dtype=np.float32)
        self.heads=np.zeros((capacity,), dtype=np.int32)
        self.head_masks=np.zeros((capacity, n_heads), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_flat_states = np.zeros((capacity,flat_state_dim), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self,
            flat_state,
            action,
            head,
            head_mask,
            reward,
            done,
            next_flat_state,
            priority):
        
        i = self.ptr
        self.flat_states[i] = flat_state
        self.actions[i] = action
        self.heads[i] = head
        self.head_masks[i]=head_mask
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
            self.heads[idxs],
            self.head_masks[idxs],
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
            self.heads[idxs],
            self.head_masks[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_flat_states[idxs],
            self.priorities[idxs],
            probs[idxs],
            idxs
        )

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities
        
#This trains, but _slowly_
def main(args):
    console.log('Setting up env and agent')
    env = gym.make("CartPole-v1")
    flat_dim=4
    action_dim=2
    n_heads=10
    agent = CartPoleAgent(
        flat_dim=flat_dim,
        action_dim=action_dim,
        n_heads=n_heads
    )
    agent.Q_prime_1_model.set_weights(agent.Q_model.get_weights())

    console.log('Running training of the Q model')
    replay_buffer = ReplayBuffer(capacity = 20000, flat_state_dim=flat_dim, action_dim=action_dim, n_heads=n_heads)

    gamma = 0.99
    tau_1 = 0.01
    tau_2 = 0.005
    beta=0.0
    beta_annealing_factor = 0.99

    head_mask_prob = 0.5

    total_episodes = 1000
    total_steps=0
    replay_buffer_warmup_period = 200
    batch_size=128
    Q_prime_delay = 200
    sums_of_rewards = []
    reward_sum_queue = []
    reward_sum_queue_length = 10

    for episode_num in track(range(total_episodes), console = console):
        start_time = time.perf_counter()
        obs, info = env.reset()

        done = False
        rewards=[]

        head = nprng.integers(0,n_heads)
        while not done:
            total_steps+=1
            Q_value, action = agent.get_max_Q_model_action(state=obs, head=head)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done=terminated or truncated
            rewards.append(int(reward))

            _, next_action = agent.get_max_Q_model_action(state=next_obs, head=head)

            Q_prime_value = agent.get_Q_prime_model_value(state=next_obs, action=next_action, head=head)
            target=reward+gamma*(1.0-done)*Q_prime_value
            td_error=float(abs(Q_value-target)[0]+1e-8)
            head_mask = nprng.binomial(1, p=head_mask_prob, size=(n_heads,))

            replay_buffer.add(
                flat_state=obs,
                action=action,
                head=head,
                head_mask=head_mask,
                reward=reward,
                done=int(done),
                next_flat_state=next_obs,
                priority=td_error,
            )

            obs = next_obs

            if total_steps >= replay_buffer_warmup_period:
                flat_states, actions, heads, head_masks, buffer_rewards, dones, next_flat_states, priorities, probs, idxs = replay_buffer.weighted_sample(batch_size=batch_size)

                _, next_actions = agent.get_max_Q_model_actions(next_flat_states, heads)
                Q_primes = agent.get_Q_prime_model_values(next_flat_states, next_actions, heads) #shape (batches,)
                target_array = buffer_rewards+(1.0-dones)*gamma*Q_primes #shape (batches,)
                final_Q_targets = np.mean(
                    agent.Q_model.predict(flat_states, verbose=0),
                    axis=-1
                )
                #shape = (examples, heads, actions)

                #Get the new td error for updating how poorly the Q network does at predicting this
                new_td_error = np.abs(final_Q_targets[np.arange(batch_size), heads, actions.astype(np.int32)]-target_array).astype(np.float32)

                #we need to figure out which heads now have updated targets
                # head_masks shape (examples, n_heads)
                # actions shape (examples, )
                #this logic should still work, since it does not involve the quantiles ever
                #and only figures out single values per head
                update_mask = (head_masks[..., None].astype(np.bool)) & (np.arange(2) == actions[:, None, None])
                final_Q_targets[update_mask] = np.tile(target_array.reshape((128,1,1)), (1,n_heads,2))[update_mask]

                bias_weights = (1.0/(replay_buffer.size*(probs)))**beta
                bias_weights/=np.max(bias_weights)
                beta = 1.0-(beta_annealing_factor**(episode_num))

                agent.Q_model.fit(
                    x=flat_states,
                    y=final_Q_targets,
                    verbose=0,
                    sample_weight = bias_weights,
                )
                replay_buffer.update_priorities(idxs, new_td_error)
                # if total_steps % Q_prime_delay == 0:
                #     agent.Q_prime_1_model.set_weights(agent.Q_model.get_weights())
                # if (total_steps+(Q_prime_delay//2)) % Q_prime_delay == 0:
                #     agent.Q_prime_2_model.set_weights(agent.Q_model.get_weights())'
                agent.ema_update(tau_1=tau_1, tau_2=tau_2)
            
        end_time = time.perf_counter()
        sums_of_rewards.append(np.sum(rewards))
        if len(reward_sum_queue) < reward_sum_queue_length:
            reward_sum_queue.append(np.sum(rewards))
        else:
            reward_sum_queue[episode_num % reward_sum_queue_length] = np.sum(rewards)
        console.log()
        console.log(f'Episode: {episode_num}, Episode Length: {end_time-start_time:.2f}s, Total Steps: {total_steps}')
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
