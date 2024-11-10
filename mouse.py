import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd

# Environment definition
class CheesePoisonMouseEnv:
    def __init__(self, grid_size=(2, 3)):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Initialize the grid and place elements
        self.grid = np.zeros(self.grid_size)

        # Place Cheese at (1, 2)
        self.cheese_pos = (1, 2)
        self.grid[self.cheese_pos] = 1  # Cheese represented by 1

        # Place Poison at (1, 1)
        self.poison_positions = [(1, 1)]
        for pos in self.poison_positions:
            self.grid[pos] = -1  # Poison represented by -1

        # Mouse starts at (0, 0)
        self.mouse_pos = (0, 0)
        return self.mouse_pos

    def step(self, action):
        # Define the movement actions
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        # Calculate new position based on the action
        move = action_map.get(action)
        new_pos = (self.mouse_pos[0] + move[0], self.mouse_pos[1] + move[1])

        # Check for boundaries
        if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
            self.mouse_pos = new_pos

        # Determine the reward based on the new position
        if self.mouse_pos == self.cheese_pos:
            reward = 10  # Reward for finding cheese
            done = True
        elif self.mouse_pos in self.poison_positions:
            reward = -10  # Penalty for stepping on poison
            done = True
        else:
            reward = -1  # Small negative reward for each move
            done = False

        return self.mouse_pos, reward, done

    def render(self):
        env = np.zeros(self.grid_size, dtype=str)
        env[:, :] = '-'

        # Place Cheese
        env[self.cheese_pos] = 'C'

        # Place Poison
        for pos in self.poison_positions:
            env[pos] = 'P'

        # Place Mouse
        env[self.mouse_pos] = 'M'

        print("\n".join(" ".join(row) for row in env))
        print()

# Define the DQN network with smaller layers
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        return self.fc2(x)  # Output Q-values for each action

# Setup the environment and hyperparameters
env = CheesePoisonMouseEnv()
state_size = env.grid_size[0] * env.grid_size[1]  # Flattened grid
action_size = 4  # Four possible actions
gamma = 0.7
epsilon = 0.1
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
num_episodes = 800

# Initialize DQN, optimizer, and loss function
dqn = DQN(input_dim=state_size, output_dim=action_size)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def preprocess_state(state):
    # Convert the (x, y) position to a one-hot encoded flattened state
    one_hot = np.zeros(state_size)
    one_hot[state[0] * env.grid_size[1] + state[1]] = 1
    return torch.FloatTensor(one_hot).unsqueeze(0)

def render_q_values_table(dqn, grid_size):
    action_symbols = ['↑', '↓', '←', '→']
    data = []

    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            state = preprocess_state((i, j))
            q_values = dqn(state).detach().numpy().flatten()
            q_value_strings = [f"{action_symbols[k]}:{q_values[k]:.2f}" for k in range(len(action_symbols))]
            row.append(" | ".join(q_value_strings))
        data.append(row)

    # Convert to a DataFrame for table-like display
    df = pd.DataFrame(data, index=[f"Row {i}" for i in range(grid_size[0])],
                      columns=[f"Col {j}" for j in range(grid_size[1])])
    print(df)

# Function to print the weights of each layer
def print_weights(dqn, episode):
    print(f"\nWeights at Episode {episode + 1}:")
    for name, param in dqn.named_parameters():
        if 'weight' in name:
            print(f"{name} weights:\n", param.data.numpy())

# Training loop without batching
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    total_reward = 0


    while not done:
        # ε-Greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(action_size))  # Explore
        else:
            with torch.no_grad():
                q_values = dqn(state)
                action = torch.argmax(q_values).item()  # Exploit

        # Take action and observe new state and reward
        next_state, reward, done = env.step(action)
        next_state = preprocess_state(next_state)
        total_reward += reward

        # Calculate target Q-value
        with torch.no_grad():
            max_next_q_value = dqn(next_state).max().item()
            target_q_value = reward + (gamma * max_next_q_value * (1 - done))

        # Update DQN: Predict Q-value and calculate loss
        q_value = dqn(state)[0, action]
        loss = loss_fn(q_value, torch.tensor(target_q_value))

        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Move to the next state
        state = next_state

        # Render the environment and Q-values after each step with a delay
        env.render()
        # time.sleep(0.2)  # Adjust delay as needed for visibility

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print weights every 10 episodes
    if (episode + 1) % 10 == 0:
        print_weights(dqn, episode)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

print("Training finished.")
render_q_values_table(dqn, env.grid_size)
