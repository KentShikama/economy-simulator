import random
from dataclasses import dataclass, field
from typing import Dict, List


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
                return f"{self.name} bought {item} from {seller.name} for {price}."
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
                return f"{self.name} sold {item} to {buyer.name} for {price}."
            return f"{self.name} cannot sell {item} because no one can afford it."
        return f"{self.name} tried to sell {item}, but there are no buyers in {self.city}."


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
                    weights[i] = max(1, 80 - self.fullness)
            elif action == "buy_apple":
                other_people_in_city_with_apple = [person for person in other_people_in_city if
                                                   person.inventory["apple"] > 0]
                if other_people_in_city_with_apple:
                    seller = min(other_people_in_city_with_apple, key=lambda x: x.prices["apple"])
                    can_afford = self.money >= seller.prices["apple"]
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        weights[i] = (100 - self.fullness)
            elif action == "sell_water":
                if self.inventory["water"] > 0:
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
                    weights[i] = max(1, 80 - self.fullness)
            elif action == "buy_apple":
                other_people_in_city_with_apple = [person for person in other_people_in_city if
                                                   person.inventory["apple"] > 0]
                if other_people_in_city_with_apple:
                    seller = min(other_people_in_city_with_apple, key=lambda x: x.prices["apple"])
                    can_afford = self.money >= seller.prices["apple"]
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        weights[i] = (100 - self.fullness)
            elif action == "sell_fertilizer":
                if self.inventory["fertilizer"] > 0:
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
class Farmer(Person):
    def act(self, other_people: List['Person']):
        actions = ['grow_apple', 'buy_water', 'buy_fertilizer', 'sell_apple', 'consume_apple']
        weights = self.build_weights(actions, other_people)
        try:
            action = random.choices(actions, weights=weights, k=1)[0]
        except ValueError:
            action = 'consume_apple'
        if action == 'grow_apple':
            return self.grow_apple()
        elif action == 'buy_water':
            return self.buy('water', other_people)
        elif action == 'buy_fertilizer':
            return self.buy('fertilizer', other_people)
        elif action == 'sell_apple':
            return self.sell('apple', other_people)
        elif action == 'consume_apple':
            return self.consume('apple')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    weights[i] = max(1, 80 - self.fullness)
            elif action == "buy_water":
                other_people_in_city_with_water = [person for person in other_people_in_city if
                                                   person.inventory["water"] > 0]
                if other_people_in_city_with_water:
                    seller = min(other_people_in_city_with_water, key=lambda x: x.prices["water"])
                    can_afford = self.money >= seller.prices["water"]
                    is_profitable = self.prices["apple"] > self.prices["fertilizer"] + seller.prices["water"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "buy_fertilizer":
                other_people_in_city_with_fertilizer = [person for person in other_people_in_city if
                                                        person.inventory["fertilizer"] > 0]
                if other_people_in_city_with_fertilizer:
                    seller = min(other_people_in_city_with_fertilizer, key=lambda x: x.prices["fertilizer"])
                    can_afford = self.money >= seller.prices["fertilizer"]
                    is_profitable = self.prices["apple"] > self.prices["water"] + seller.prices["fertilizer"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "sell_apple":
                if self.inventory["apple"] > 0:
                    weights[i] = (self.inventory["apple"] // 10) + 1
            elif action == "grow_apple":
                if self.inventory["water"] > 0 and self.inventory["fertilizer"] > 0:
                    weights[i] = 1
        return weights

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
                        weights[i] = 1
            elif 'move' in action:
                city = action.split('_')[1]
                if city != self.city:
                    weights[i] = 1
        return weights

    def move(self, city):
        self.city = city
        return f"{self.name} moved to {self.city}."


class EconomySimulator:
    def __init__(self):
        self.people = [
            WaterCollector(name="Digger", city="A", money=100),
            FertilizerCreator(name="Dirt", city="B", money=100),
            Farmer(name="Farmer Joe", city="C", money=100),
            Peddler(name="Carrier X", city="A", money=100),
            Peddler(name="Carrier Y", city="B", money=100),
        ]
        self.day = 0
        self.is_running = False
        self.action_log = []

    def simulate_day(self):
        self.day += 1
        day_actions = []
        
        for person in self.people:
            if person.fullness <= 0:
                raise Exception(f"{person.name} has died of hunger.")
            person.fullness -= 1
            result = person.act([p for p in self.people if p != person])
            day_actions.append({
                'person': person.name,
                'city': person.city,
                'money': person.money,
                'fullness': person.fullness,
                'prices': dict(person.prices),
                'inventory': dict(person.inventory),
                'action': result
            })
        
        self.action_log.append({
            'day': self.day,
            'actions': day_actions
        })
        
        return day_actions

    def reset(self):
        self.__init__()