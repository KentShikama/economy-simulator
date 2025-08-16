import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from grid import Grid

# Constant for fullness addition when consuming items
FULLNESS_ADDITION = 20

# Price adjustment constants
PRICE_ADJUSTMENT_RATE = 0.05

# Peddler inventory limit
MAX_INVENTORY_PEDDLER = 50


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
    grid_x: int = 0
    grid_y: int = 0
    fullness: int = 100
    inventory: Dict[str, int] = field(default_factory=lambda: {'seed': 0, 'fertilizer': 0, 'grain': 40})
    prices: Dict[str, float] = field(default_factory=lambda: {'seed': 10, 'fertilizer': 10, 'grain': 10})

    def update_position(self, grid: Grid):
        """Update the person's current city based on their grid position"""
        self.city = grid.get_city_at(self.grid_x, self.grid_y)

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
                self.prices['seed'] *= (1 + PRICE_ADJUSTMENT_RATE)
                self.prices['fertilizer'] *= (1 + PRICE_ADJUSTMENT_RATE)

        if self.fullness < 70:
            self.prices['grain'] *= (1 + PRICE_ADJUSTMENT_RATE)

    def buy(self, item, other_people: List['Person']):
        # People can trade if they're in the same city (including None for traveling)
        sellers = [person for person in other_people if
                   person.inventory[item] > 0 and person.city == self.city and self.city is not None]
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

        # Can only sell if in a city
        if self.city is None:
            return f"{self.name} must be in a city to sell {item}."

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
class SeedCollector(Person):
    def act(self, other_people: List['Person']):
        actions = ['collect_seed', 'sell_seed', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        action = random.choices(actions, weights=weights, k=1)[0]
        if action == 'collect_seed':
            return self.collect_seed()
        elif action == 'sell_seed':
            return self.sell('seed', other_people)
        elif action == 'buy_grain':
            return self.buy('grain', other_people)
        elif action == 'consume_grain':
            return self.consume('grain')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_grain":
                if self.inventory["grain"] > 0:
                    weights[i] = max(0, 80 - self.fullness, 9999 if self.fullness < 50 else 0)
            elif action == "buy_grain":
                other_people_in_city_with_grain = [person for person in other_people_in_city if
                                                   person.inventory["grain"] > 0]
                if other_people_in_city_with_grain:
                    seller = min(other_people_in_city_with_grain, key=lambda x: x.prices["grain"])
                    can_afford = self.money >= seller.prices["grain"]
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        weights[i] = (100 - self.fullness)
            elif action == "sell_seed":
                if self.inventory["seed"] > 0:
                    weights[i] = max(10, (self.inventory["seed"] // 10) + 1)
            elif action == "collect_seed":
                weights[i] = 1
        return weights

    def collect_seed(self):
        self.inventory['seed'] += 10
        message = f"{self.name} collected 10 units of seed."
        return ActionResult(ActionType.COLLECT, 'seed', message)


@dataclass
class FertilizerCreator(Person):
    def act(self, other_people: List['Person']):
        actions = ['produce_fertilizer', 'sell_fertilizer', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        action = random.choices(actions, weights=weights, k=1)[0]
        if action == 'produce_fertilizer':
            return self.produce_fertilizer()
        elif action == 'sell_fertilizer':
            return self.sell('fertilizer', other_people)
        elif action == 'buy_grain':
            return self.buy('grain', other_people)
        elif action == 'consume_grain':
            return self.consume('grain')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_grain":
                if self.inventory["grain"] > 0:
                    weights[i] = max(0, 80 - self.fullness, 9999 if self.fullness < 50 else 0)
            elif action == "buy_grain":
                other_people_in_city_with_grain = [person for person in other_people_in_city if
                                                   person.inventory["grain"] > 0]
                if other_people_in_city_with_grain:
                    seller = min(other_people_in_city_with_grain, key=lambda x: x.prices["grain"])
                    can_afford = self.money >= seller.prices["grain"]
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
        actions = ['grow_grain', 'buy_seed', 'buy_fertilizer', 'sell_grain', 'consume_grain', 'do_nothing']
        weights = self.build_weights(actions, other_people)
        try:
            action = random.choices(actions, weights=weights, k=1)[0]
        except ValueError:
            action = 'consume_apple'
        if action == 'grow_grain':
            return self.grow_grain()
        elif action == 'buy_seed':
            return self.buy('seed', other_people)
        elif action == 'buy_fertilizer':
            return self.buy('fertilizer', other_people)
        elif action == 'sell_grain':
            return self.sell('grain', other_people)
        elif action == 'consume_grain':
            return self.consume('grain')
        elif action == 'do_nothing':
            return f"{self.name} is resting."
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city]
        for i, action in enumerate(actions):
            if action == "consume_grain":
                if self.inventory["grain"] > 0:
                    weights[i] = max(0, 80 - self.fullness, 9999 if self.fullness < 50 else 0)
            elif action == "buy_seed":
                other_people_in_city_with_seed = [person for person in other_people_in_city if
                                                   person.inventory["seed"] > 0]
                if other_people_in_city_with_seed:
                    seller = min(other_people_in_city_with_seed, key=lambda x: x.prices["seed"])
                    can_afford = self.money >= seller.prices["seed"]
                    is_profitable = self.prices["grain"] > self.prices["fertilizer"] + seller.prices["seed"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "buy_fertilizer":
                other_people_in_city_with_fertilizer = [person for person in other_people_in_city if
                                                        person.inventory["fertilizer"] > 0]
                if other_people_in_city_with_fertilizer:
                    seller = min(other_people_in_city_with_fertilizer, key=lambda x: x.prices["fertilizer"])
                    can_afford = self.money >= seller.prices["fertilizer"]
                    is_profitable = self.prices["grain"] > self.prices["seed"] + seller.prices["fertilizer"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "sell_grain":
                if self.inventory["grain"] > 0:
                    weights[i] = (self.inventory["grain"] // 10) + 1
            elif action == "grow_grain":
                if self.inventory["seed"] > 0 and self.inventory["fertilizer"] > 0:
                    weights[i] = 1
            elif action == "do_nothing":
                weights[i] = 1
        return weights

    def grow_grain(self):
        if self.inventory['seed'] > 0 and self.inventory['fertilizer'] > 0:
            self.inventory['seed'] -= 1
            self.inventory['fertilizer'] -= 1
            self.inventory['grain'] += 10
            message = f"{self.name} grew 10 units of grain."
            return ActionResult(ActionType.GROW, 'grain', message)
        return f"{self.name} does not have enough resources to grow grain."


@dataclass
class Peddler(Person):
    target_city: Optional[str] = None

    def act(self, other_people: List['Person'], grid: Grid):
        actions = ['move_to_Seeds', 'move_to_Mulch', 'move_to_Harvest', 'buy_seed', 'buy_fertilizer', 'buy_grain',
                   'sell_seed', 'sell_fertilizer', 'sell_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        try:
            action = random.choices(actions, weights=weights, k=1)[0]
        except ValueError:
            action = 'do_nothing'

        if 'move_to_' in action:
            destination_city = action.split('_')[-1]
            return self.move_toward_target(grid, destination_city)
        elif 'buy' in action:
            item = action.split('_')[1]
            return self.buy(item, other_people)
        elif 'sell' in action:
            item = action.split('_')[1]
            return self.sell(item, other_people)
        elif action == 'consume_grain':
            return self.consume('grain')
        return "Invalid action"

    def build_weights(self, actions, other_people):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city and self.city is not None]

        for i, action in enumerate(actions):
            if action == "consume_grain":
                if self.inventory["grain"] > 0:
                    weights[i] = max(0, 80 - self.fullness, 9999 if self.fullness < 50 else 0)
            elif 'buy' in action and self.city is not None:  # Can only buy in cities
                item = action.split('_')[1]
                other_people_in_city_with_item = [person for person in other_people_in_city if
                                                  person.inventory[item] > 0]
                if other_people_in_city_with_item and self.inventory[item] < MAX_INVENTORY_PEDDLER:
                    seller = min(other_people_in_city_with_item, key=lambda x: x.prices[item])
                    can_afford = self.money >= seller.prices[item]
                    avg_price = sum([person.prices[item] for person in other_people]) / len(other_people) if other_people else seller.prices[item]
                    profit_margin = (avg_price - seller.prices[item]) / seller.prices[item] if seller.prices[item] > 0 else 0
                    if can_afford and profit_margin > 0:
                        weights[i] = max(1, int(profit_margin * 10))
            elif 'sell' in action and self.city is not None:  # Can only sell in cities
                item = action.split('_')[1]
                if self.inventory[item] > 0:
                    buyer_candidates = [person for person in other_people_in_city]
                    if buyer_candidates:
                        buyer = max(buyer_candidates, key=lambda x: x.prices[item])
                        avg_price = sum([person.prices[item] for person in other_people]) / len(other_people) if other_people else buyer.prices[item]
                        profit_margin = (buyer.prices[item] - avg_price) / avg_price if avg_price > 0 else 0
                        if profit_margin > 0:
                            weights[i] = max(1, int(profit_margin * 10))
            elif 'move_to_' in action:
                destination_city = action.split('_')[-1]

                # Don't move to the city we're already in
                if destination_city == self.city:
                    weights[i] = 0
                    continue

                # If we already have this city as our target, continue toward it
                if self.target_city == destination_city:
                    weights[i] = 1
                else:
                    # Basic weight for new destinations when not currently traveling
                    if self.target_city is None:
                        weights[i] = 1
                    else:
                        # No changing destination mid-journey
                        weights[i] = 0

        return weights

    def move_toward_target(self, grid: Grid, destination_city: str = None):
        """Move one step toward the target city"""
        # Set new target if provided
        if destination_city:
            # Check if already at destination
            if destination_city == self.city:
                return f"{self.name} is already in {destination_city}."
            self.target_city = destination_city

        if not self.target_city:
            return f"{self.name} has no destination."

        # Get path from current position to target
        path = grid.find_path_to_city(self.grid_x, self.grid_y, self.target_city)

        if not path:
            # Already at destination or no path found
            if self.city == self.target_city:
                self.target_city = None
                return f"{self.name} has arrived at {self.city}."
            else:
                return f"{self.name} cannot find a path to {self.target_city}."

        # Move to next position (first step in path)
        next_x, next_y = path[0]
        old_city = self.city
        self.grid_x = next_x
        self.grid_y = next_y
        self.update_position(grid)

        # Check if we entered or left a city
        if old_city != self.city:
            if self.city:
                # Clear target when we reach the destination city
                if self.city == self.target_city:
                    self.target_city = None
                return f"{self.name} entered {self.city}."
            else:
                return f"{self.name} left {old_city}."
        else:
            # Calculate direction for descriptive message
            old_x, old_y = self.grid_x - (next_x - self.grid_x), self.grid_y - (next_y - self.grid_y)
            if next_x > old_x:
                direction = "right"
            elif next_x < old_x:
                direction = "left"
            elif next_y > old_y:
                direction = "down"
            else:
                direction = "up"

            return f"{self.name} moved {direction} toward {self.target_city}."
