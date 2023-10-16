import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import DQN
from state import State

MAX_MEMORY = 100000
LENGTH = 20

class Agent:
    def __init__(self):
        self.num_episodes = 4000
        self.state_dim = LENGTH  # Dimensionality of the state
        self.action_dim = LENGTH + 1  # Number of possible actions
        self.learning_rate = 0.002
        self.gamma = 0.85
        self.epsilon = self.num_episodes * 2 / 3
        self.batch_size = 512
        self.replay_buffer = deque(maxlen=MAX_MEMORY)  # Replay buffer size
        self.model = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)



    def get_state(self, state):
        return np.array(state.A)
    
    def get_action(self, state, episode_number):
        # Epsilon-greedy action selection or exploration strategy
        if random.uniform(0, episode_number + self.epsilon) < self.epsilon:
            action = random.choice([i for i in range(LENGTH)])
        else:
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()
        return action

    def train_model(self, episode_number):
        key = random.choice([i for i in range(LENGTH)])
        env = State(LENGTH, key)
        state = self.get_state(env)
        done = False
    
        while not done:
            state = self.get_state(env)
            action = self.get_action(state, episode_number)

            reward, done = env.step(action)
            next_state = self.get_state(env)
        
            self.replay_buffer.append((state, action, reward, next_state, done))
        
            # Training step
            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states, dtype=float)
                next_states = np.array(next_states, dtype=float)
                actions = np.array(actions, dtype=int)
                rewards = np.array(rewards, dtype=float)
                dones = np.array(dones, dtype=float)

                states = torch.tensor(states, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

            # Calculate Q-values for the current state and next state
                q_values = self.model(states)
                q_values_next = self.model(next_states)
            
            # Q-learning update
                target_values = rewards + self.gamma * (1 - dones) * torch.max(q_values_next, dim=1).values
                predicted_values = q_values[range(self.batch_size), actions]
            
                loss = torch.mean((target_values - predicted_values) ** 2)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return env.num_steps

    def train(self):
        for i in range(self.num_episodes):
            print("Starting Episode: ", i)
            self.train_model(i)

    def test(self):
        test_mean = 0
        total_correct = 0
        for j in range(500):
            key = random.choice([i for i in range(LENGTH)])
            env = State(LENGTH, key)
            state = self.get_state(env)
            done = False
            actions = []
        
            while not done:
                state = self.get_state(env)
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()            
                actions.append(action)
                reward, done = env.step(action)

            print("Searching for: ", key)
            print(env.A)
            print(actions)
            test_mean += env.num_steps
            total_correct += done and (env.num_steps <= 0.4 * env.n)
        print("The test mean was: ", test_mean / 500)
        print("Accuracy of algorithm is: ", total_correct / 5, "%")

if __name__ == "__main__":
    a = Agent()
    a.train()
    a.test()