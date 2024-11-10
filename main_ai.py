import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network with multiple action heads
class FarmerPolicyNetwork(nn.Module):
    def __init__(self):
        super(FarmerPolicyNetwork, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(11, 128),  # Input size matches the state vector length
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Price Setting Head (Continuous Outputs)
        self.price_head = nn.Sequential(
            nn.Linear(64, 3),   # Outputs for prices (apple, water, fertilizer)
            nn.Sigmoid(),       # Normalize outputs between 0 and 1
        )
        # Action Selection Head (Discrete Outputs)
        self.action_head = nn.Sequential(
            nn.Linear(64, 5),   # Number of possible actions (grow, buy water, buy fertilizer, sell, consume)
            nn.Softmax(dim=-1), # Convert outputs to probabilities
        )

    def forward(self, x):
        shared_output = self.shared_fc(x)
        prices = self.price_head(shared_output)
        action_probs = self.action_head(shared_output)
        return prices, action_probs


@dataclass
class Person:
    name: str
    city: str
    money: float
    fullness: int = 100
    inventory: Dict[str, int] = field(default_factory=lambda: {'water': 0, 'fertilizer': 0, 'apple': 10})
    prices: Dict[str, float] = field(default_factory=lambda: {'water': 1, 'fertilizer': 1, 'apple': 1})

    def consume(self, item):
        if self.inventory[item] > 0:
            self.inventory[item] -= 1
            self.fullness = min(100, self.fullness + 20)
            return f"{self.name} consumed {item}."
        return f"{self.name} has no {item} to consume."

    def buy(self, item, other_people: List['Person']):
        sellers = [person for person in other_people if person.inventory[item] > 0 and person.city == self.city]
        if sellers:
            seller = min(sellers, key=lambda x: x.prices[item])
            price = seller.prices[item]
            if price > self.prices[item]:
                self.prices[item] *= 1.05
                return f"{self.name} refuses to buy {item} from {seller.name} because the price is too high."
            elif self.money >= price:
                self.money -= price
                seller.money += price
                self.prices[item] *= 0.95
                seller.prices[item] *= 1.05
                self.inventory[item] += 1
                seller.inventory[item] -= 1
                return f"{self.name} bought {item} from {seller.name} for ${round(price, 2)}."
            else:
                return f"{self.name} cannot afford {item}."
        return f"{self.name} tried to buy {item}, but it is out of stock."

    def sell(self, item, other_people: List['Person']):
        if self.inventory[item] <= 0:
            return f"{self.name} has no {item} to sell."

        buyers = [person for person in other_people if person.city == self.city]
        if buyers:
            buyer = max(buyers, key=lambda x: x.prices[item])
            price = buyer.prices[item]
            if price < self.prices[item]:
                self.prices[item] *= 0.95
                return f"{self.name} refuses to sell {item} to {buyer.name} because the price is too low."
            if buyer.money >= price:
                self.money += price
                buyer.money -= price
                self.prices[item] *= 1.05
                buyer.prices[item] *= 0.95
                self.inventory[item] -= 1
                buyer.inventory[item] += 1
                return f"{self.name} sold {item} to {buyer.name} for ${round(price, 2)}."
            return f"{self.name} cannot sell {item} because no one can afford it."
        return f"{self.name} tried to sell {item}, but there are no buyers in {self.city}."


# Redefine the Farmer class to include the reward function and training
@dataclass
class Farmer(Person):
    def __post_init__(self):
        # Initialize the neural network and optimizer
        self.policy_network = FarmerPolicyNetwork()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        # Initialize any other necessary variables
        self.reset_state()
        # For storing experiences
        self.log_probs = []
        self.rewards = []

    def reset_state(self):
        # Reset the Farmer's state but keep the policy network
        self.city = "C"
        self.money = 100
        self.fullness = 100
        self.inventory = {'water': 0, 'fertilizer': 0, 'apple': 10}
        self.prices = {'water': 1, 'fertilizer': 1, 'apple': 1}
        self.previous_money = self.money
        self.previous_fullness = self.fullness
        self.previous_prices = self.prices.copy()

    def get_total_circulating_money(self, people):
        return sum(person.money for person in people)

    def act(self, other_people: List['Person']):
        # Prepare the state vector
        state = self.prepare_state(other_people)
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Get prices and action probabilities from the neural network
        prices, action_probs = self.policy_network(state_tensor)
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        # Store the log probability for training
        log_prob = action_dist.log_prob(action)
        # Convert prices to actual values
        total_money = self.get_total_circulating_money([self] + other_people)
        normalized_prices = prices.detach().numpy()[0]
        new_prices = normalized_prices * total_money
        # Update Farmer's prices
        self.update_prices(new_prices)
        # Map the action index to an action string
        actions = ['grow_apple', 'buy_water', 'buy_fertilizer', 'sell_apple', 'consume_apple']
        selected_action = actions[action.item()]
        # Perform the selected action
        if selected_action == 'grow_apple':
            result = self.grow_apple()
        elif selected_action == 'buy_water':
            result = self.buy('water', other_people)
        elif selected_action == 'buy_fertilizer':
            result = self.buy('fertilizer', other_people)
        elif selected_action == 'sell_apple':
            result = self.sell('apple', other_people)
        elif selected_action == 'consume_apple':
            result = self.consume('apple')
        else:
            result = "Invalid action"

        # Calculate reward
        reward = self.calculate_reward()
        # Perform training immediately
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update previous money and fullness
        self.previous_money = self.money
        self.previous_fullness = self.fullness
        self.previous_prices = self.prices.copy()

        return result

    def calculate_reward(self):
        if self.fullness <= 0:
            return -1000  # Harsh penalty for starving

        # Survival Incentive
        reward = 1  # Small reward for staying alive

        # Fullness Maintenance
        fullness_reward = (self.fullness - 50) / 50  # Normalize between 0 and 1
        reward += fullness_reward

        # Profitability
        profit = self.money - self.previous_money
        profit_reward = profit / (self.previous_money + 1e-6)  # Avoid division by zero
        reward += profit_reward

        return reward

    def prepare_state(self, other_people):
        # Log-transform inventories
        log_inventory_apple = np.log(self.inventory['apple'] + 1)
        log_inventory_water = np.log(self.inventory['water'] + 1)
        log_inventory_fertilizer = np.log(self.inventory['fertilizer'] + 1)

        # Calculate total circulating money
        total_money = self.get_total_circulating_money([self] + other_people)
        if total_money <= 0:
            total_money = 1.0  # Avoid division by zero

        # Normalize money
        normalized_money = self.money / total_money

        # Scale fullness between 0 and 1
        normalized_fullness = self.fullness / 100.0

        # Normalize own prices
        normalized_price_apple = self.prices['apple'] / total_money
        normalized_price_water = self.prices['water'] / total_money
        normalized_price_fertilizer = self.prices['fertilizer'] / total_money

        # Normalize market minimum prices
        market_min_price_apple = min(
            [p.prices['apple'] for p in other_people if p.inventory['apple'] > 0 and p.city == self.city],
            default=self.prices['apple']
        )
        market_min_price_water = min(
            [p.prices['water'] for p in other_people if p.inventory['water'] > 0 and p.city == self.city],
            default=self.prices['water']
        )
        market_min_price_fertilizer = min(
            [p.prices['fertilizer'] for p in other_people if p.inventory['fertilizer'] > 0 and p.city == self.city],
            default=self.prices['fertilizer']
        )

        normalized_market_min_price_apple = market_min_price_apple / total_money
        normalized_market_min_price_water = market_min_price_water / total_money
        normalized_market_min_price_fertilizer = market_min_price_fertilizer / total_money

        # Construct the state vector
        state = [
            log_inventory_apple,
            log_inventory_water,
            log_inventory_fertilizer,
            normalized_money,
            normalized_fullness,
            normalized_price_apple,
            normalized_price_water,
            normalized_price_fertilizer,
            normalized_market_min_price_apple,
            normalized_market_min_price_water,
            normalized_market_min_price_fertilizer,
        ]

        return state

    def update_prices(self, new_prices):
        # Ensure prices are non-negative and update
        self.prices['apple'] = max(new_prices[0], 0.01)
        self.prices['water'] = max(new_prices[1], 0.01)
        self.prices['fertilizer'] = max(new_prices[2], 0.01)

    def grow_apple(self):
        if self.inventory['water'] > 0 and self.inventory['fertilizer'] > 0:
            self.inventory['water'] -= 1
            self.inventory['fertilizer'] -= 1
            self.inventory['apple'] += 10
            self.prices['water'] *= 1.05
            self.prices['fertilizer'] *= 1.05
            self.prices['apple'] *= 0.95
            return f"{self.name} grew 10 units of apple."
        return f"{self.name} does not have enough resources to grow apples."


@dataclass
class WaterCollector(Person):
    def act(self, other_people: List['Person']):
        actions = ['collect_water', 'sell_water', 'buy_apple', 'consume_apple']
        weights = self.build_weights(actions, other_people)
        action = random.choices(actions, weights=weights, k=1)[0]
        if action == 'collect_water':
            return self.collect_water()
        elif action == 'sell_water':
            return self.sell('water', other_people)
        elif action == 'buy_apple':
            return self.buy('apple', other_people)
        elif action == 'consume_apple':
            return self.consume('apple')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    print(f"{self.name} has a chance of consuming an apple.")
                    weights[i] = max(1, 80 - self.fullness)
            elif action == "buy_apple":
                other_people_in_city_with_apple = [person for person in other_people_in_city if
                                                   person.inventory["apple"] > 0]
                if other_people_in_city_with_apple:
                    seller = min(other_people_in_city_with_apple, key=lambda x: x.prices["apple"])
                    can_afford = self.money >= seller.prices["apple"]
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        print(f"{self.name} has a chance of buying an apple.")
                        weights[i] = (100 - self.fullness)
            elif action == "sell_water":
                if self.inventory["water"] > 0:
                    print(f"{self.name} has a chance of selling water.")
                    weights[i] = (self.inventory["water"] // 10) + 1
            elif action == "collect_water":
                if self.inventory["water"] == 0:
                    weights[i] = 1
        return weights

    def collect_water(self):
        self.inventory['water'] += 10
        self.prices['water'] *= 0.95
        return f"{self.name} collected 10 units of water."


@dataclass
class FertilizerCreator(Person):
    def act(self, other_people: List['Person']):
        actions = ['produce_fertilizer', 'sell_fertilizer', 'buy_apple', 'consume_apple']
        weights = self.build_weights(actions, other_people)
        action = random.choices(actions, weights=weights, k=1)[0]
        if action == 'produce_fertilizer':
            return self.produce_fertilizer()
        elif action == 'sell_fertilizer':
            return self.sell('fertilizer', other_people)
        elif action == 'buy_apple':
            return self.buy('apple', other_people)
        elif action == 'consume_apple':
            return self.consume('apple')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    print(f"{self.name} has a chance of consuming an apple.")
                    weights[i] = max(1, 80 - self.fullness)
            elif action == "buy_apple":
                other_people_in_city_with_apple = [person for person in other_people_in_city if
                                                   person.inventory["apple"] > 0]
                if other_people_in_city_with_apple:
                    seller = min(other_people_in_city_with_apple, key=lambda x: x.prices["apple"])
                    can_afford = self.money >= seller.prices["apple"]
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        print(f"{self.name} has a chance of buying an apple.")
                        weights[i] = (100 - self.fullness)
            elif action == "sell_fertilizer":
                if self.inventory["fertilizer"] > 0:
                    print(f"{self.name} has a chance of selling fertilizer.")
                    weights[i] = (self.inventory["fertilizer"] // 10) + 1
            elif action == "produce_fertilizer":
                if self.inventory["fertilizer"] == 0:
                    weights[i] = 1
        return weights

    def produce_fertilizer(self):
        self.inventory['fertilizer'] += 10
        self.prices['fertilizer'] *= 0.95
        return f"{self.name} produced 10 units of fertilizer."

@dataclass
class Peddler(Person):
    def act(self, other_people: List['Person']):
        actions = ['move_A', 'move_B', 'move_C', 'buy_water', 'buy_fertilizer', 'buy_apple', 'sell_water',
                   'sell_fertilizer', 'sell_apple', 'consume_apple']
        weights = self.build_weights(actions, other_people)
        try:
            action = random.choices(actions, weights=weights, k=1)[0]
        except ValueError:
            action = random.choice(['move_A', 'move_B', 'move_C'])
        if 'move' in action:
            city = action.split('_')[1]
            return self.move(city)
        elif 'buy' in action:
            item = action.split('_')[1]
            return self.buy(item, other_people)
        elif 'sell' in action:
            item = action.split('_')[1]
            return self.sell(item, other_people)
        elif action == 'consume_apple':
            return self.consume('apple')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        other_people_in_other_cities = [person for person in other_people if person.city != self.city]
        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    print(f"{self.name} has a chance of consuming an apple.")
                    weights[i] = max(1, 80 - self.fullness)
            elif 'buy' in action:
                item = action.split('_')[1]
                other_people_in_city_with_item = [person for person in other_people_in_city if
                                                  person.inventory[item] > 0]
                if other_people_in_city_with_item:
                    seller = min(other_people_in_city_with_item, key=lambda x: x.prices[item])
                    can_afford = self.money >= seller.prices[item]
                    is_profitable = seller.prices[item] < (
                            sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                        other_people_in_other_cities))
                    if can_afford and is_profitable:
                        print(f"{self.name} has a chance of buying {item}.")
                        weights[i] = 1
            elif 'sell' in action:
                item = action.split('_')[1]
                other_people_in_city = [person for person in other_people if person.city == self.city]
                if other_people_in_city and self.inventory[item] > 0:
                    buyer = max(other_people_in_city, key=lambda x: x.prices[item])
                    is_profitable = buyer.prices[item] > (
                            sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                        other_people_in_other_cities))
                    if is_profitable:
                        print(f"{self.name} has a chance of selling {item}.")
                        weights[i] = 1
            elif 'move' in action:
                city = action.split('_')[1]
                if city != self.city:
                    print(f"{self.name} has a chance of moving to {city}.")
                    weights[i] = 1
        return weights

    def move(self, city):
        self.city = city
        return f"{self.name} moved to {self.city}."


# Create the Farmer instance (only once)
farmer = Farmer(name="Farmer Joe", city="C", money=100)

# Run the simulation for 1000 episodes
for episode in range(1, 101):
    print(f"Episode {episode}")
    # Initialize other people
    people = [
        WaterCollector(name="Digger", city="A", money=100),
        FertilizerCreator(name="Dirt", city="B", money=100),
        farmer,  # Use the same Farmer instance
        Peddler(name="Carrier X", city="A", money=100),
        Peddler(name="Carrier Y", city="B", money=100),
    ]
    # Reset the Farmer's state
    farmer.reset_state()
    day = 0
    while True:
        day += 1
        # Simulate a day
        print(f"Day {day}")
        for person in people[:]:  # Use a copy of the list to avoid modification issues
            if person.fullness <= 0:
                print(f"{person.name} has died of hunger.")
                print(f"Episode ended at day {day}.")
                break  # End the episode if someone dies
            person.fullness -= 1
            if person == farmer:
                result = person.act([p for p in people if p != person])
            else:
                result = person.act([p for p in people if p != person])
            rounded_prices = {k: f"${round(v, 2)}" for k, v in person.prices.items()}
            print(
                f"City: {person.city:<2} Money: ${int(person.money):<4} Fullness: {person.fullness:<3} Prices: {rounded_prices} Inventory: {person.inventory} Action: {result}"
            )
        else:
            continue  # Continue if the inner loop wasn't broken
        break  # Break if the inner loop was broken (Farmer died)
    print(f"End of Episode {episode}\n")
