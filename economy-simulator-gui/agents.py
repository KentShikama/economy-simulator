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
MAX_INVENTORY_PEDDLER = 100

# Trade unit size - all trades must be in multiples of this amount
TRADE_UNIT_SIZE = 10

# Grain production yield from 1 seed + 1 fertilizer
GRAIN_YIELD = 10

# Survival grain reserve - agents try to keep this much grain
SURVIVAL_RESERVE = 100

# Production amounts for collectors/creators
SEED_COLLECTION_AMOUNT = 10
FERTILIZER_PRODUCTION_AMOUNT = 10


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
    inventory: Dict[str, int] = field(default_factory=lambda: {'seed': 0, 'fertilizer': 0, 'grain': 50})
    prices: Dict[str, float] = field(default_factory=lambda: {'seed': 10, 'fertilizer': 10, 'grain': 10})
    latest_weights: Dict[str, float] = field(default_factory=dict)
    target_city: Optional[str] = None

    def update_position(self, grid: Grid):
        """Update the person's current city based on their grid position"""
        self.city = grid.get_city_at(self.grid_x, self.grid_y)
    
    def move_toward_target(self, grid: Grid, destination_city: str = None):
        """Move one step toward the target city"""
        if destination_city:
            if destination_city == self.city:
                return f"{self.name} is already in {destination_city}."
            self.target_city = destination_city
        
        if not self.target_city:
            return f"{self.name} has no destination."
        
        path = grid.find_path_to_city(self.grid_x, self.grid_y, self.target_city)
        
        if not path:
            if self.city == self.target_city:
                self.target_city = None
                return f"{self.name} has arrived at {self.city}."
            else:
                return f"{self.name} cannot find a path to {self.target_city}."
        
        next_x, next_y = path[0]
        old_city = self.city
        old_x, old_y = self.grid_x, self.grid_y
        self.grid_x, self.grid_y = next_x, next_y
        self.update_position(grid)
        
        if old_city != self.city:
            if self.city:
                if self.city == self.target_city:
                    self.target_city = None
                return f"{self.name} entered {self.city}."
            else:
                return f"{self.name} left {old_city}."
        else:
            if next_x > old_x:
                direction = "right"
            elif next_x < old_x:
                direction = "left"
            elif next_y > old_y:
                direction = "down"
            elif next_y < old_y:
                direction = "up"
            else:
                raise Exception(f"{self.name} didn't move but path exists: old=({old_x},{old_y}) next=({next_x},{next_y})")
            return f"{self.name} moved {direction} toward {self.target_city}."

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
                   person.inventory[item] >= TRADE_UNIT_SIZE and person.city == self.city and self.city is not None]
        if sellers:
            seller = min(sellers, key=lambda x: x.prices[item])
            price = seller.prices[item] * TRADE_UNIT_SIZE
            if seller.prices[item] > self.prices[item]:
                message = f"{self.name} refuses to buy {TRADE_UNIT_SIZE} {item} from {seller.name} because the price is too high."
                return ActionResult(ActionType.BUY_REFUSED, item, message, seller)
            elif self.money >= price:
                self.money -= price
                seller.money += price
                self.inventory[item] += TRADE_UNIT_SIZE
                seller.inventory[item] -= TRADE_UNIT_SIZE
                message = f"{self.name} bought {TRADE_UNIT_SIZE} {item} from {seller.name} for {price}."
                return ActionResult(ActionType.BUY_SUCCESS, item, message, seller)
            else:
                message = f"{self.name} cannot afford {TRADE_UNIT_SIZE} {item}."
                return ActionResult(ActionType.BUY_REFUSED, item, message, seller)
        message = f"{self.name} tried to buy {TRADE_UNIT_SIZE} {item}, but insufficient stock available (need {TRADE_UNIT_SIZE}+ units)."
        return ActionResult(ActionType.BUY_REFUSED, item, message, None)

    def sell(self, item, other_people: List['Person']):
        if self.inventory[item] < TRADE_UNIT_SIZE:
            return f"{self.name} has insufficient {item} to sell (need {TRADE_UNIT_SIZE}+ units, have {self.inventory[item]})."

        # Can only sell if in a city
        if self.city is None:
            return f"{self.name} must be in a city to sell {item}."

        buyers = [person for person in other_people if person.city == self.city]
        if buyers:
            buyer = max(buyers, key=lambda x: x.prices[item])
            price = self.prices[item] * TRADE_UNIT_SIZE
            if buyer.prices[item] < self.prices[item]:
                message = f"{self.name} refuses to sell {TRADE_UNIT_SIZE} {item} to {buyer.name} because the price is too low."
                return ActionResult(ActionType.SELL_REFUSED, item, message, buyer)
            if buyer.money >= price:
                self.money += price
                buyer.money -= price
                self.inventory[item] -= TRADE_UNIT_SIZE
                buyer.inventory[item] += TRADE_UNIT_SIZE
                message = f"{self.name} sold {TRADE_UNIT_SIZE} {item} to {buyer.name} for {price}."
                return ActionResult(ActionType.SELL_SUCCESS, item, message, buyer)
            message = f"{self.name} cannot sell {TRADE_UNIT_SIZE} {item} because no one can afford it."
            return ActionResult(ActionType.SELL_REFUSED, item, message, buyer)
        message = f"{self.name} tried to sell {TRADE_UNIT_SIZE} {item}, but there are no buyers in {self.city}."
        return ActionResult(ActionType.SELL_REFUSED, item, message, None)

    def can_sell_profitably(self, item, other_people_in_city):
        """Check if we can sell an item profitably in current city"""
        if self.inventory[item] >= TRADE_UNIT_SIZE and other_people_in_city:
            buyer = max(other_people_in_city, key=lambda x: x.prices[item])
            profit_margin = (buyer.prices[item] - self.prices[item]) * TRADE_UNIT_SIZE
            return profit_margin > 0
        return False
    
    def can_buy_profitably(self, item, other_people_in_city):
        """Check if we can buy an item profitably in current city"""
        sellers = [p for p in other_people_in_city if p.inventory[item] >= TRADE_UNIT_SIZE]
        if sellers:
            seller = min(sellers, key=lambda x: x.prices[item])
            if self.money >= seller.prices[item] * TRADE_UNIT_SIZE:
                profit_margin = (self.prices[item] - seller.prices[item]) * TRADE_UNIT_SIZE
                return profit_margin > 0
        return False
    
    def calculate_grain_survival_weight(self, other_people_in_city):
        """Calculate weight for buying grain based on survival needs"""
        sellers_with_grain = [p for p in other_people_in_city if p.inventory["grain"] >= TRADE_UNIT_SIZE]
        if sellers_with_grain:
            seller = min(sellers_with_grain, key=lambda x: x.prices["grain"])
            can_afford = self.money >= seller.prices["grain"] * TRADE_UNIT_SIZE
            if can_afford:
                grain_deficit = max(0, SURVIVAL_RESERVE - self.inventory["grain"])
                return grain_deficit // TRADE_UNIT_SIZE
        return 0


@dataclass
class SeedCollector(Person):
    def act(self, other_people: List['Person']):
        actions = ['collect_seed', 'sell_seed', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
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
                other_people_in_city_with_grain = [person for person in other_people_in_city if person.inventory["grain"] >= TRADE_UNIT_SIZE]
                if other_people_in_city_with_grain:
                    seller = min(other_people_in_city_with_grain, key=lambda x: x.prices["grain"])
                    can_afford = self.money >= seller.prices["grain"] * TRADE_UNIT_SIZE
                    is_not_full = self.fullness < 100
                    if can_afford and is_not_full:
                        weights[i] = (100 - self.fullness)
            elif action == "sell_seed":
                if self.inventory["seed"] >= TRADE_UNIT_SIZE:
                    weights[i] = 1  # TODO: Think harder
            elif action == "collect_seed":
                weights[i] = 1
        return weights

    def collect_seed(self):
        self.inventory['seed'] += SEED_COLLECTION_AMOUNT
        message = f"{self.name} collected {SEED_COLLECTION_AMOUNT} units of seed."
        return ActionResult(ActionType.COLLECT, 'seed', message)


@dataclass
class FertilizerCreator(Person):
    def act(self, other_people: List['Person']):
        actions = ['produce_fertilizer', 'sell_fertilizer', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
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
                                                   person.inventory["grain"] >= TRADE_UNIT_SIZE]
                if other_people_in_city_with_grain:
                    seller = min(other_people_in_city_with_grain, key=lambda x: x.prices["grain"])
                    can_afford = self.money >= seller.prices["grain"] * TRADE_UNIT_SIZE
                    if can_afford:
                        # Buy grain if below survival reserve
                        grain_deficit = max(0, SURVIVAL_RESERVE - self.inventory["grain"])
                        weights[i] = grain_deficit // TRADE_UNIT_SIZE
            elif action == "sell_fertilizer":
                if self.inventory["fertilizer"] >= TRADE_UNIT_SIZE:
                    weights[i] = 1  # TODO: Think harder
            elif action == "produce_fertilizer":
                weights[i] = 1
        return weights

    def produce_fertilizer(self):
        self.inventory['fertilizer'] += FERTILIZER_PRODUCTION_AMOUNT
        message = f"{self.name} produced {FERTILIZER_PRODUCTION_AMOUNT} units of fertilizer."
        return ActionResult(ActionType.PRODUCE, 'fertilizer', message)


@dataclass
class Farmer(Person):
    def act(self, other_people: List['Person']):
        actions = ['grow_grain', 'buy_seed', 'buy_fertilizer', 'sell_grain', 'consume_grain', 'do_nothing']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
        try:
            action = random.choices(actions, weights=weights, k=1)[0]
        except ValueError:
            action = 'do_nothing'
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
                                                   person.inventory["seed"] >= TRADE_UNIT_SIZE]
                if other_people_in_city_with_seed:
                    seller = min(other_people_in_city_with_seed, key=lambda x: x.prices["seed"])
                    can_afford = self.money >= seller.prices["seed"] * TRADE_UNIT_SIZE
                    is_profitable = GRAIN_YIELD * self.prices["grain"] > self.prices["fertilizer"] + seller.prices["seed"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "buy_fertilizer":
                other_people_in_city_with_fertilizer = [person for person in other_people_in_city if
                                                        person.inventory["fertilizer"] >= TRADE_UNIT_SIZE]
                if other_people_in_city_with_fertilizer:
                    seller = min(other_people_in_city_with_fertilizer, key=lambda x: x.prices["fertilizer"])
                    can_afford = self.money >= seller.prices["fertilizer"] * TRADE_UNIT_SIZE
                    is_profitable = GRAIN_YIELD * self.prices["grain"] > self.prices["seed"] + seller.prices["fertilizer"]
                    if can_afford and is_profitable:
                        weights[i] = 1
            elif action == "sell_grain":
                if self.inventory["grain"] >= TRADE_UNIT_SIZE:
                    survival_reserve = SURVIVAL_RESERVE
                    sellable_units = max(0, (self.inventory["grain"] - survival_reserve) // TRADE_UNIT_SIZE)
                    weights[i] = sellable_units
            elif action == "grow_grain":
                if self.inventory["seed"] > 0 and self.inventory["fertilizer"] > 0:
                    weights[i] = 1
        return weights

    def grow_grain(self):
        if self.inventory['seed'] > 0 and self.inventory['fertilizer'] > 0:
            self.inventory['seed'] -= 1
            self.inventory['fertilizer'] -= 1
            self.inventory['grain'] += GRAIN_YIELD
            message = f"{self.name} grew {GRAIN_YIELD} units of grain."
            return ActionResult(ActionType.GROW, 'grain', message)
        return f"{self.name} does not have enough resources to grow grain."


class SeedPeddler(Person):
    """Peddler that specializes in trading seeds between Seeds and Harvest cities"""
    
    def act(self, other_people: List['Person'], grid: Grid):
        # Limited actions: can only move between Seeds and Harvest, trade seeds, and buy grain
        actions = ['move_to_Seeds', 'move_to_Harvest', 'buy_seed', 'sell_seed', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
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
            elif action == 'buy_seed':
                if self.city == 'Seeds' and self.can_buy_profitably('seed', other_people_in_city):
                    weights[i] = 1
            elif action == 'sell_seed':
                if self.city == 'Harvest' and self.can_sell_profitably('seed', other_people_in_city):
                    weights[i] = 1
            elif action == "buy_grain":
                weights[i] = self.calculate_grain_survival_weight(other_people_in_city)
            elif action == 'move_to_Seeds':
                if self.city != 'Seeds':
                    if self.target_city == 'Seeds':
                        weights[i] = 1  # Continue toward target
                    elif self.target_city is None and not self.can_sell_profitably('seed', other_people_in_city):
                        weights[i] = 1  # Go buy seeds if we can't sell profitably anymore
            elif action == 'move_to_Harvest':
                if self.city != 'Harvest':
                    if self.target_city == 'Harvest':
                        weights[i] = 1  # Continue toward target
                    elif self.target_city is None and not self.can_buy_profitably('seed', other_people_in_city):
                        weights[i] = 1  # Go sell seeds if we can't buy profitably anymore
        return weights


class FertilizerPeddler(Person):
    """Peddler that specializes in trading fertilizer between Mulch and Harvest cities"""
    
    def act(self, other_people: List['Person'], grid: Grid):
        # Limited actions: can only move between Mulch and Harvest, trade fertilizer, and buy grain
        actions = ['move_to_Mulch', 'move_to_Harvest', 'buy_fertilizer', 'sell_fertilizer', 'buy_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
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
            elif action == 'buy_fertilizer':
                if self.city == 'Mulch' and self.can_buy_profitably('fertilizer', other_people_in_city):
                    weights[i] = 1
            elif action == 'sell_fertilizer':
                if self.city == 'Harvest' and self.can_sell_profitably('fertilizer', other_people_in_city):
                    weights[i] = 1
            elif action == "buy_grain":
                weights[i] = self.calculate_grain_survival_weight(other_people_in_city)
            elif action == 'move_to_Mulch':
                if self.city != 'Mulch':
                    if self.target_city == 'Mulch':
                        weights[i] = 1  # Continue toward target
                    elif self.target_city is None and not self.can_sell_profitably('fertilizer', other_people_in_city):
                        weights[i] = 1  # Go buy fertilizer if we can't sell profitably anymore
            elif action == 'move_to_Harvest':
                if self.city != 'Harvest':
                    if self.target_city == 'Harvest':
                        weights[i] = 1  # Continue toward target
                    elif self.target_city is None and not self.can_buy_profitably('fertilizer', other_people_in_city):
                        weights[i] = 1  # Go sell fertilizer if we can't buy profitably anymore
        return weights


class GrainPeddler(Person):
    """Peddler that specializes in trading grain between all cities"""
    
    def calculate_grain_profit_weight(self, other_people_in_city):
        """Calculate weight for buying grain based on profit potential"""
        if self.can_buy_profitably("grain", other_people_in_city):
            return 1
        return 0
    
    def act(self, other_people: List['Person'], grid: Grid):
        # Can move to all cities but only trades grain
        actions = ['move_to_Seeds', 'move_to_Mulch', 'move_to_Harvest', 'buy_grain', 'sell_grain', 'consume_grain']
        weights = self.build_weights(actions, other_people)
        self.latest_weights = dict(zip(actions, weights))
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
            elif action == 'buy_grain' and self.city is not None:
                survival_weight = self.calculate_grain_survival_weight(other_people_in_city)
                profit_weight = self.calculate_grain_profit_weight(other_people_in_city)
                weights[i] = max(survival_weight, profit_weight)
            elif action == 'sell_grain' and self.city is not None:
                if self.can_sell_profitably('grain', other_people_in_city):
                    # Only sell if we have more than survival reserve
                    sellable_units = max(0, (self.inventory["grain"] - SURVIVAL_RESERVE) // TRADE_UNIT_SIZE)
                    weights[i] = sellable_units
            elif 'move_to_' in action:
                destination_city = action.split('_')[-1]
                if destination_city == self.city:
                    weights[i] = 0
                elif self.target_city == destination_city:
                    weights[i] = 1  # Continue toward target
                elif self.target_city is None:
                    # Evaluate profit potential for destination
                    people_in_dest = [p for p in other_people if p.city == destination_city]
                    if people_in_dest:
                        # Check if we can buy or sell profitably there
                        can_buy_prof = self.can_buy_profitably('grain', people_in_dest)
                        can_sell_prof = self.can_sell_profitably('grain', people_in_dest)
                        if can_buy_prof or can_sell_prof:
                            weights[i] = 1
        return weights
