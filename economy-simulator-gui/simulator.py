import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

# Constant for fullness addition when consuming items
FULLNESS_ADDITION = 20

# Price adjustment constants
PRICE_ADJUSTMENT_RATE = 0.05

# Peddler inventory limit
MAX_INVENTORY_PEDDLER = 20

# Grid dimensions
GRID_WIDTH = 30
GRID_HEIGHT = 20

# City dimensions (each city is 5x5)
CITY_SIZE = 5


class CellType(Enum):
    PATH = "path"
    BLOCKED = "blocked"
    CITY = "city"


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
class Grid:
    width: int
    height: int
    cells: List[List[CellType]] = field(default_factory=list)
    city_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize grid with all paths
        self.cells = [[CellType.PATH for _ in range(self.width)] for _ in range(self.height)]

        # Place cities (each city is 5x5)
        # City A at top-left area
        self.city_positions['A'] = (3, 3)
        # City B at top-right area
        self.city_positions['B'] = (22, 3)
        # City C at bottom-center area
        self.city_positions['C'] = (12, 13)

        # Mark city cells
        for city_name, (cx, cy) in self.city_positions.items():
            for dy in range(CITY_SIZE):
                for dx in range(CITY_SIZE):
                    if 0 <= cx + dx < self.width and 0 <= cy + dy < self.height:
                        self.cells[cy + dy][cx + dx] = CellType.CITY

        # Add some blocked cells around the map for obstacles
        # Create some natural barriers
        blocked_regions = [
            # Mountains between cities
            (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
            (15, 9), (16, 9), (17, 9),
            # Forest areas
            (1, 10), (2, 10), (1, 11), (2, 11),
            (26, 10), (27, 10), (28, 10), (26, 11), (27, 11),
        ]

        for bx, by in blocked_regions:
            if 0 <= bx < self.width and 0 <= by < self.height:
                # Don't block city cells
                if self.cells[by][bx] != CellType.CITY:
                    self.cells[by][bx] = CellType.BLOCKED

    def get_city_at(self, x: int, y: int) -> Optional[str]:
        """Returns the city name if the position is within a city, None otherwise"""
        for city_name, (cx, cy) in self.city_positions.items():
            if cx <= x < cx + CITY_SIZE and cy <= y < cy + CITY_SIZE:
                return city_name
        return None

    def is_valid_move(self, x: int, y: int) -> bool:
        """Check if a position is valid for movement"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.cells[y][x] != CellType.BLOCKED

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (up, down, left, right)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_move(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    def find_path_to_city(self, start_x: int, start_y: int, target_city: str) -> List[Tuple[int, int]]:
        """Find shortest path from start position to target city using BFS"""
        if target_city not in self.city_positions:
            return []

        # Get city center as target
        cx, cy = self.city_positions[target_city]
        target_x, target_y = cx + 2, cy + 2  # Center of 5x5 city

        # If already at target
        if start_x == target_x and start_y == target_y:
            return []

        # BFS to find shortest path
        queue = deque([(start_x, start_y, [])])
        visited = {(start_x, start_y)}

        while queue:
            x, y, path = queue.popleft()

            # Check all neighbors
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]

                    # Check if we reached the target city
                    if self.get_city_at(nx, ny) == target_city:
                        return new_path

                    queue.append((nx, ny, new_path))

        return []  # No path found


@dataclass
class Person:
    name: str
    city: str
    money: float
    grid_x: int = 0
    grid_y: int = 0
    destination_x: Optional[int] = None
    destination_y: Optional[int] = None
    fullness: int = 100
    inventory: Dict[str, int] = field(default_factory=lambda: {'water': 0, 'fertilizer': 0, 'apple': 10})
    prices: Dict[str, float] = field(default_factory=lambda: {'water': 1, 'fertilizer': 1, 'apple': 1})

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
                self.prices['water'] *= (1 + PRICE_ADJUSTMENT_RATE)
                self.prices['fertilizer'] *= (1 + PRICE_ADJUSTMENT_RATE)

        if self.fullness < 70:
            self.prices['apple'] *= (1 + PRICE_ADJUSTMENT_RATE / 5)

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
    target_city: Optional[str] = None

    def act(self, other_people: List['Person'], grid: Grid):
        actions = ['move_to_A', 'move_to_B', 'move_to_C', 'buy_water', 'buy_fertilizer', 'buy_apple',
                   'sell_water', 'sell_fertilizer', 'sell_apple', 'consume_apple', 'do_nothing']
        weights = self.build_weights(actions, other_people, grid)
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
        elif action == 'consume_apple':
            return self.consume('apple')
        return "Invalid action"

    def build_weights(self, actions, other_people, grid: Grid):
        weights = [0 for _action in actions]
        other_people_in_city = [person for person in other_people if person.city == self.city and self.city is not None]
        other_people_in_other_cities = [person for person in other_people if
                                        person.city != self.city and person.city is not None]

        for i, action in enumerate(actions):
            if action == "consume_apple":
                if self.inventory["apple"] > 0:
                    weights[i] = max(0, 80 - self.fullness)
            elif 'buy' in action and self.city is not None:  # Can only buy in cities
                item = action.split('_')[1]
                other_people_in_city_with_item = [person for person in other_people_in_city if
                                                  person.inventory[item] > 0]
                if other_people_in_city_with_item and self.inventory[item] < MAX_INVENTORY_PEDDLER:
                    seller = min(other_people_in_city_with_item, key=lambda x: x.prices[item])
                    can_afford = self.money >= seller.prices[item]
                    if other_people_in_other_cities:
                        avg_price = sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                            other_people_in_other_cities)
                        profit_margin = (avg_price - seller.prices[item]) / seller.prices[item] if seller.prices[
                                                                                                       item] > 0 else 0
                        if can_afford and profit_margin > 0:
                            weights[i] = max(1, int(profit_margin * 10))
            elif 'sell' in action and self.city is not None:  # Can only sell in cities
                item = action.split('_')[1]
                if self.inventory[item] > 0:
                    buyer_candidates = [person for person in other_people_in_city]
                    if buyer_candidates:
                        buyer = max(buyer_candidates, key=lambda x: x.prices[item])
                        if other_people_in_other_cities:
                            avg_price = sum([person.prices[item] for person in other_people_in_other_cities]) / len(
                                other_people_in_other_cities)
                            profit_margin = (buyer.prices[item] - avg_price) / avg_price if avg_price > 0 else 0
                            if profit_margin > 0:
                                weights[i] = max(1, int(profit_margin * 10))
            elif 'move_to_' in action:
                destination_city = action.split('_')[-1]

                # Don't move to the city we're already in
                if destination_city == self.city:
                    weights[i] = 0
                    continue

                # If we already have this city as our target, heavily weight continuing toward it
                if self.target_city == destination_city:
                    weights[i] = 10  # High weight for persistence
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


class EconomySimulator:
    def __init__(self):
        self.grid = Grid(GRID_WIDTH, GRID_HEIGHT)

        # Initialize people with grid positions
        self.people = [
            WaterCollector(name="Digger", city="A", money=100),
            FertilizerCreator(name="Dirt", city="B", money=100),
            Farmer(name="Farmer Joe", city="C", money=100),
            Peddler(name="Carrier X", city="A", money=100),
            Peddler(name="Carrier Y", city="B", money=100),
        ]

        # Place people in their starting cities (center of each city)
        for person in self.people:
            if person.city in self.grid.city_positions:
                cx, cy = self.grid.city_positions[person.city]
                person.grid_x = cx + 2  # Center of 5x5 city
                person.grid_y = cy + 2

        self.day = 0
        self.is_running = False
        self.action_log = []
        self.ticks_per_day = 20
        self.tick_count = 0

    def tick(self):
        # Update positions based on grid
        for person in self.people:
            person.update_position(self.grid)

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
            # Peddlers pass the grid for movement
            if isinstance(person, Peddler):
                result = person.act([p for p in self.people if p != person], self.grid)
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
