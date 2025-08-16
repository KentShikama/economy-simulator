import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# Constant for fullness addition when consuming items
FULLNESS_ADDITION = 20

# Price adjustment constants
PRICE_ADJUSTMENT_RATE = 0.05

# Peddler inventory limit
MAX_INVENTORY_PEDDLER = 20


class ActionType(Enum):
    BUY_SUCCESS = "buy_success"
    BUY_REFUSED = "buy_refused"
    SELL_SUCCESS = "sell_success"
    SELL_REFUSED = "sell_refused"
    PRODUCE = "produce"
    COLLECT = "collect"
    GROW = "grow"


@dataclass
class ActionResult:
    action_type: ActionType
    item: str
    message: str
    other_person: Optional['Person'] = None


@dataclass
class Person:
    name: str
    city: str
    money: float
    x: float = 0.0
    y: float = 0.0
    destination: str = None
    speed: float = 5.0
    fullness: int = 100
    inventory: Dict[str, int] = field(default_factory=lambda: {'water': 0, 'fertilizer': 0, 'apple': 10})
    prices: Dict[str, float] = field(default_factory=lambda: {'water': 1, 'fertilizer': 1, 'apple': 1})

    def update_movement(self, cities):
        if self.destination:
            dest_city_coords = cities[self.destination]
            dx = dest_city_coords['x'] - self.x
            dy = dest_city_coords['y'] - self.y
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance < self.speed:
                self.x = dest_city_coords['x']
                self.y = dest_city_coords['y']
                self.city = self.destination
                self.destination = None
            else:
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed

    def consume(self, item):
        if self.inventory[item] > 0:
            self.inventory[item] -= 1
            self.fullness = min(100, self.fullness + FULLNESS_ADDITION)
            return f"{self.name} consumed {item}."
        return f"{self.name} has no {item} to consume."

    def adjust_prices(self, action_result: ActionResult):
        if isinstance(action_result, ActionResult):
            action_type = action_result.action_type
            item = action_result.item
            other_person = action_result.other_person

            if action_type == ActionType.BUY_SUCCESS:
                self.prices[item] *= (1 - PRICE_ADJUSTMENT_RATE)
                other_person.prices[item] *= (1 + PRICE_ADJUSTMENT_RATE)
            elif action_type == ActionType.BUY_REFUSED:
                self.prices[item] *= (1 + PRICE_ADJUSTMENT_RATE)
            elif action_type == ActionType.SELL_SUCCESS:
                self.prices[item] *= (1 + PRICE_ADJUSTMENT_RATE)
                other_person.prices[item] *= (1 - PRICE_ADJUSTMENT_RATE)
            elif action_type == ActionType.SELL_REFUSED:
                self.prices[item] *= (1 - PRICE_ADJUSTMENT_RATE)
            elif action_type in [ActionType.PRODUCE, ActionType.COLLECT]:
                self.prices[item] *= (1 - PRICE_ADJUSTMENT_RATE)
            elif action_type == ActionType.GROW:
                self.prices[item] *= (1 - PRICE_ADJUSTMENT_RATE)
                self.prices['water'] *= (1 + PRICE_ADJUSTMENT_RATE)
                self.prices['fertilizer'] *= (1 + PRICE_ADJUSTMENT_RATE)

        if self.fullness < 70:
            self.prices['apple'] *= (1 + PRICE_ADJUSTMENT_RATE / 5)

    def buy(self, item, other_people: List['Person']):
        sellers = [person for person in other_people if person.inventory[item] > 0 and person.city == self.city]
        if sellers:
            seller = min(sellers, key=lambda x: x.prices[item])
            price = seller.prices[item]
            if price > self.prices[item]:
                message = f"{self.name} refuses to buy {item} from {seller.name} because the price is too high."
                return ActionResult(ActionType.BUY_REFUSED, item, message, seller)
            elif self.money >= price:
                self.money -= price
                seller.money += price
                self.inventory[item] += 1
                seller.inventory[item] -= 1
                message = f"{self.name} bought {item} from {seller.name} for {price}."
                return ActionResult(ActionType.BUY_SUCCESS, item, message, seller)
            else:
                message = f"{self.name} cannot afford {item}."
                return ActionResult(ActionType.BUY_REFUSED, item, message, seller)
        message = f"{self.name} tried to buy {item}, but it is out of stock."
        return ActionResult(ActionType.BUY_REFUSED, item, message, None)

    def sell(self, item, other_people: List['Person']):
        if self.inventory[item] <= 0:
            return f"{self.name} has no {item} to sell."

        buyers = [person for person in other_people if person.city == self.city]
        if buyers:
            buyer = max(buyers, key=lambda x: x.prices[item])
            price = buyer.prices[item]
            if price < self.prices[item]:
                message = f"{self.name} refuses to sell {item} to {buyer.name} because the price is too low."
                return ActionResult(ActionType.SELL_REFUSED, item, message, buyer)
            if buyer.money >= price:
                self.money += price
                buyer.money -= price
                self.inventory[item] -= 1
                buyer.inventory[item] += 1
                message = f"{self.name} sold {item} to {buyer.name} for {price}."
                return ActionResult(ActionType.SELL_SUCCESS, item, message, buyer)
            message = f"{self.name} cannot sell {item} because no one can afford it."
            return ActionResult(ActionType.SELL_REFUSED, item, message, buyer)
        message = f"{self.name} tried to sell {item}, but there are no buyers in {self.city}."
        return ActionResult(ActionType.SELL_REFUSED, item, message, None)


@dataclass
class WaterCollector(Person):
    def act(self, other_people: List['Person']):
        actions = ['collect_water', 'sell_water', 'buy_apple', 'consume_apple', 'do_nothing']
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
                    weights[i] = max(0, 80 - self.fullness)
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
                    weights[i] = max(10, (self.inventory["water"] // 10) + 1)
            elif action == "collect_water":
                weights[i] = 1
        return weights

    def collect_water(self):
        self.inventory['water'] += 10
        message = f"{self.name} collected 10 units of water."
        return ActionResult(ActionType.COLLECT, 'water', message)


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
                    weights[i] = max(0, 80 - self.fullness)
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
                    weights[i] = max(10, (self.inventory["fertilizer"] // 10) + 1)
            elif action == "produce_fertilizer":
                weights[i] = 1
        return weights

    def produce_fertilizer(self):
        self.inventory['fertilizer'] += 10
        message = f"{self.name} produced 10 units of fertilizer."
        return ActionResult(ActionType.PRODUCE, 'fertilizer', message)


@dataclass
class Farmer(Person):
    def act(self, other_people: List['Person']):
        actions = ['grow_apple', 'buy_water', 'buy_fertilizer', 'sell_apple', 'consume_apple', 'do_nothing']
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
        elif action == 'do_nothing':
            return f"{self.name} is resting."
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    weights[i] = max(0, 80 - self.fullness)
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
            elif action == "do_nothing":
                weights[i] = 1
        return weights

    def grow_apple(self):
        if self.inventory['water'] > 0 and self.inventory['fertilizer'] > 0:
            self.inventory['water'] -= 1
            self.inventory['fertilizer'] -= 1
            self.inventory['apple'] += 10
            message = f"{self.name} grew 10 units of apple."
            return ActionResult(ActionType.GROW, 'apple', message)
        return f"{self.name} does not have enough resources to grow apples."


@dataclass
class Peddler(Person):
    def act(self, other_people: List['Person']):
        actions = ['move_A', 'move_B', 'move_C', 'buy_water', 'buy_fertilizer', 'buy_apple', 'sell_water',
                   'sell_fertilizer', 'sell_apple', 'consume_apple', 'do_nothing']
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
                    weights[i] = max(0, 80 - self.fullness)
            elif 'buy' in action:
                item = action.split('_')[1]
                other_people_in_city_with_item = [person for person in other_people_in_city if
                                                  person.inventory[item] > 0]
                if other_people_in_city_with_item and self.inventory[item] < MAX_INVENTORY_PEDDLER:
                    seller = min(other_people_in_city_with_item, key=lambda x: x.prices[item])
                    can_afford = self.money >= seller.prices[item]
                    avg_price = sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                        other_people_in_other_cities)
                    profit_margin = (avg_price - seller.prices[item]) / seller.prices[item]
                    if can_afford and profit_margin > 0:
                        weights[i] = max(1, int(profit_margin * 10))
            elif 'sell' in action:
                item = action.split('_')[1]
                other_people_in_city = [person for person in other_people if person.city == self.city]
                if other_people_in_city and self.inventory[item] > 0:
                    buyer = max(other_people_in_city, key=lambda x: x.prices[item])
                    avg_price = sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                        other_people_in_other_cities)
                    profit_margin = (buyer.prices[item] - avg_price) / avg_price
                    if profit_margin > 0:
                        weights[i] = max(1, int(profit_margin * 10))
            elif 'move' in action:
                city = action.split('_')[1]
                if city != self.city and not self.destination:
                    weights[i] = 1
        return weights

    def move(self, city):
        if self.city != city:
            self.destination = city
            self.city = None
            return f"{self.name} started moving to {city}."
        return f"{self.name} is already in {city}."


class EconomySimulator:
    def __init__(self):
        self.cities = {
            'A': {'x': 860, 'y': 150},
            'B': {'x': 1280, 'y': 150},
            'C': {'x': 1070, 'y': 380}
        }
        self.people = [
            WaterCollector(name="Digger", city="A", money=100),
            FertilizerCreator(name="Dirt", city="B", money=100),
            Farmer(name="Farmer Joe", city="C", money=100),
            Peddler(name="Carrier X", city="A", money=100),
            Peddler(name="Carrier Y", city="B", money=100),
        ]
        for person in self.people:
            if person.city in self.cities:
                person.x = self.cities[person.city]['x']
                person.y = self.cities[person.city]['y']

        self.day = 0
        self.is_running = False
        self.action_log = []
        self.ticks_per_day = 20
        self.tick_count = 0

    def tick(self):
        for person in self.people:
            person.update_movement(self.cities)

        self.tick_count += 1
        if self.tick_count >= self.ticks_per_day:
            self.tick_count = 0
            return self.simulate_day()

        return []

    def simulate_day(self):
        self.day += 1
        day_actions = []

        for person in self.people:
            if person.fullness <= 0:
                raise Exception(f"{person.name} has died of hunger.")
            person.fullness -= 1

            result = None
            if person.destination:  # Don't act while moving but they can still consume
                if person.fullness < 25 and person.inventory['apple'] > 0:
                    result = person.consume('apple')
            else:
                result = person.act([p for p in self.people if p != person])
                person.adjust_prices(result)

            if result:
                action_text = result.message if isinstance(result, ActionResult) else result
                day_actions.append({
                    'person': person.name,
                    'city': person.city,
                    'money': person.money,
                    'fullness': person.fullness,
                    'prices': dict(person.prices),
                    'inventory': dict(person.inventory),
                    'action': action_text
                })

        if day_actions:
            self.action_log.append({
                'day': self.day,
                'actions': day_actions
            })

        return day_actions

    def reset(self):
        self.__init__()


def main():
    """Run the simulator in text-only mode"""
    print("Economy Simulator - Text Mode")
    print("=" * 50)

    simulator = EconomySimulator()

    try:
        while True:
            # Run one full day (20 ticks)
            for _ in range(simulator.ticks_per_day):
                actions = simulator.tick()

                # Print actions if it's the end of a day
                if actions:
                    print(f"\nDay {simulator.day}:")
                    print("-" * 30)
                    for action in actions:
                        print(f"  {action['action']}")
                        print(
                            f"    City: {action['city']}, Money: ${action['money']:.2f}, Fullness: {action['fullness']}%")

            # Add a small delay to make it readable
            import time
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
        print(f"Final day: {simulator.day}")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print(f"Simulation ended on day {simulator.day}")


if __name__ == "__main__":
    main()
