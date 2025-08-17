from agents import SeedCollector, FertilizerCreator, Farmer, Peddler, ActionResult
from grid import Grid, GRID_WIDTH, GRID_HEIGHT
import json
import os
from datetime import datetime


class EconomySimulator:
    def __init__(self):
        self.grid = Grid(GRID_WIDTH, GRID_HEIGHT)

        # Initialize people with grid positions
        # City of Seeds: Seed Collectors
        # City of Mulch: Fertilizer Creators  
        # City of Harvest: Farmers
        # Peddlers: Travel between cities
        self.people = [
            SeedCollector(name="Harvester Harry", city="Seeds", money=1000),
            FertilizerCreator(name="Composter Carl", city="Mulch", money=1000),
            Farmer(name="Farmer Frank", city="Harvest", money=1000),
            Peddler(name="Peddler Pete", city="Seeds", money=1000),
            Peddler(name="Peddler Penny", city="Mulch", money=1000),
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
        
        # Logging setup - always enabled
        self.log_file = "economy_log.json"
        self._init_log_file()

    def _init_log_file(self):
        """Initialize the log file with metadata"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
            "initial_agents": [
                {
                    "name": person.name,
                    "type": type(person).__name__,
                    "city": person.city,
                    "initial_money": person.money,
                    "grid_x": person.grid_x,
                    "grid_y": person.grid_y
                }
                for person in self.people
            ]
        }
        
        # Write metadata as first line
        with open(self.log_file, 'w') as f:
            f.write(json.dumps({"metadata": metadata}) + '\n')

    def _append_to_log(self, day_data):
        """Append daily data to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(day_data) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")

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
            day_log = {
                'day': self.day,
                'actions': day_actions
            }
            self.action_log.append(day_log)
            
            # Log to file if enabled
            self._append_to_log(day_log)

        return day_actions

    def reset(self):
        self.__init__()


def main():
    """Run the simulator in text-only mode"""
    print("Economy Simulator - Text Mode")
    print("=" * 50)
    print("Logging enabled - data will be saved to economy_log.json")

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
