import pygame
import sys
import math
from simulator import EconomySimulator, CellType, GRID_WIDTH, GRID_HEIGHT

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
BROWN = (139, 69, 19)
RED = (231, 76, 60)
ORANGE = (230, 126, 34)
LIGHT_GRAY = (240, 240, 240)
CITY_COLOR = (255, 245, 200)  # Light yellow for cities
BLOCKED_COLOR = (80, 80, 80)  # Dark gray for blocked
PATH_COLOR = (220, 220, 220)  # Light gray for paths

# Person type colors
TYPE_COLORS = {
    'WaterCollector': BLUE,
    'FertilizerCreator': BROWN,
    'Farmer': GREEN,
    'Peddler': RED
}


class Button:
    def __init__(self, x, y, width, height, text, color=GRAY, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 24)
        self.is_hovered = False

    def draw(self, screen):
        color = self.color if not self.is_hovered else tuple(min(255, c + 30) for c in self.color)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class PersonCard:
    def __init__(self, x, y, person):
        self.x = x
        self.y = y
        self.person = person
        self.width = 220
        self.height = 240
        self.font_title = pygame.font.Font(None, 18)
        self.font_text = pygame.font.Font(None, 14)

    def draw(self, screen):
        # Background
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)

        # Type header with color
        type_name = type(self.person).__name__
        type_color = TYPE_COLORS.get(type_name, BLACK)
        pygame.draw.rect(screen, type_color, (self.x, self.y, self.width, 30))

        # Name
        name_surface = self.font_title.render(self.person.name, True, WHITE)
        screen.blit(name_surface, (self.x + 10, self.y + 5))

        y_offset = self.y + 40

        # City and Money (show position if not in city)
        if self.person.city:
            city_text = f"City: {self.person.city}"
        else:
            city_text = f"Pos: ({self.person.grid_x}, {self.person.grid_y})"
        money_text = f"Money: ${self.person.money:.2f}"

        screen.blit(self.font_text.render(city_text, True, BLACK), (self.x + 10, y_offset))
        screen.blit(self.font_text.render(money_text, True, BLACK), (self.x + 110, y_offset))
        y_offset += 20

        # Fullness bar
        screen.blit(self.font_text.render("Fullness:", True, BLACK), (self.x + 10, y_offset))
        y_offset += 20

        # Draw fullness bar
        bar_width = 180
        bar_height = 15
        bar_x = self.x + 10
        bar_y = y_offset

        # Background bar
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_width, bar_height))

        # Fullness bar with color coding
        fullness_color = GREEN if self.person.fullness > 50 else ORANGE if self.person.fullness > 20 else RED
        filled_width = int((self.person.fullness / 100) * bar_width)
        pygame.draw.rect(screen, fullness_color, (bar_x, bar_y, filled_width, bar_height))
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)

        # Fullness text
        fullness_text = self.font_text.render(f"{self.person.fullness}%", True, BLACK)
        screen.blit(fullness_text, (bar_x + bar_width - 35, bar_y + 2))

        y_offset += 25

        # Inventory
        screen.blit(self.font_text.render("Inventory:", True, BLACK), (self.x + 10, y_offset))
        y_offset += 15

        for item, count in self.person.inventory.items():
            text = f"{item}: {count}"
            screen.blit(self.font_text.render(text, True, BLACK), (self.x + 20, y_offset))
            y_offset += 15

        y_offset += 5

        # Prices
        screen.blit(self.font_text.render("Prices:", True, BLACK), (self.x + 10, y_offset))
        y_offset += 15

        for item, price in self.person.prices.items():
            text = f"{item}: ${price:.2f}"
            screen.blit(self.font_text.render(text, True, BLACK), (self.x + 20, y_offset))
            y_offset += 15


class MapView:
    def __init__(self, x, y, width, height, simulator):
        self.rect = pygame.Rect(x, y, width, height)
        self.simulator = simulator
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 12)
        self.tiny_font = pygame.font.Font(None, 10)
        
        # Calculate cell size based on grid dimensions
        self.cell_size = min(
            (width - 20) // GRID_WIDTH,
            (height - 20) // GRID_HEIGHT
        )
        
        # Center the grid in the view
        grid_width = self.cell_size * GRID_WIDTH
        grid_height = self.cell_size * GRID_HEIGHT
        self.grid_x = x + (width - grid_width) // 2
        self.grid_y = y + (height - grid_height) // 2

    def draw(self, screen):
        # Draw map background
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw grid cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell_rect = pygame.Rect(
                    self.grid_x + x * self.cell_size,
                    self.grid_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Determine cell color
                cell_type = self.simulator.grid.cells[y][x]
                if cell_type == CellType.CITY:
                    color = CITY_COLOR
                elif cell_type == CellType.BLOCKED:
                    color = BLOCKED_COLOR
                else:
                    color = PATH_COLOR
                
                pygame.draw.rect(screen, color, cell_rect)
                pygame.draw.rect(screen, GRAY, cell_rect, 1)
        
        # Draw city labels
        for city_name, (cx, cy) in self.simulator.grid.city_positions.items():
            # Calculate center of city
            center_x = self.grid_x + (cx + 2) * self.cell_size + self.cell_size // 2
            center_y = self.grid_y + (cy + 2) * self.cell_size + self.cell_size // 2
            
            # Draw city name
            text = self.font.render(f"City {city_name}", True, BLACK)
            text_rect = text.get_rect(center=(center_x, center_y - self.cell_size * 2))
            screen.blit(text, text_rect)
            
            # Draw city border
            city_rect = pygame.Rect(
                self.grid_x + cx * self.cell_size,
                self.grid_y + cy * self.cell_size,
                5 * self.cell_size,
                5 * self.cell_size
            )
            pygame.draw.rect(screen, BLACK, city_rect, 2)
        
        # Draw people
        people_by_pos = {}
        for person in self.simulator.people:
            pos_key = (person.grid_x, person.grid_y)
            if pos_key not in people_by_pos:
                people_by_pos[pos_key] = []
            people_by_pos[pos_key].append(person)
        
        for (gx, gy), people in people_by_pos.items():
            # Calculate screen position
            screen_x = self.grid_x + gx * self.cell_size + self.cell_size // 2
            screen_y = self.grid_y + gy * self.cell_size + self.cell_size // 2
            
            if len(people) == 1:
                # Single person in cell
                person = people[0]
                type_name = type(person).__name__
                color = TYPE_COLORS.get(type_name, BLACK)
                
                radius = min(self.cell_size // 3, 8)
                pygame.draw.circle(screen, color, (screen_x, screen_y), radius)
                pygame.draw.circle(screen, BLACK, (screen_x, screen_y), radius, 1)
                
                # Draw name below if space permits
                if self.cell_size > 15:
                    name_text = self.tiny_font.render(person.name[:8], True, BLACK)
                    text_rect = name_text.get_rect(centerx=screen_x, top=screen_y + radius + 2)
                    screen.blit(name_text, text_rect)
            else:
                # Multiple people in cell - arrange in small circle
                num_people = len(people)
                small_radius = min(self.cell_size // 4, 5)
                
                for i, person in enumerate(people):
                    angle = (2 * math.pi * i) / num_people
                    offset_x = int(small_radius * math.cos(angle))
                    offset_y = int(small_radius * math.sin(angle))
                    
                    type_name = type(person).__name__
                    color = TYPE_COLORS.get(type_name, BLACK)
                    
                    person_radius = min(self.cell_size // 6, 4)
                    pygame.draw.circle(screen, color, 
                                     (screen_x + offset_x, screen_y + offset_y), 
                                     person_radius)
                    pygame.draw.circle(screen, BLACK, 
                                     (screen_x + offset_x, screen_y + offset_y), 
                                     person_radius, 1)


class EconomySimulatorGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Economy Simulator")
        self.clock = pygame.time.Clock()

        self.simulator = EconomySimulator()
        self.running = True
        self.paused = True
        self.last_update_time = pygame.time.get_ticks()
        self.days_per_second = 1.0  # Simulation speed
        self.day_accumulator = 0.0

        # UI elements
        self.play_button = Button(20, WINDOW_HEIGHT - 60, 100, 40, "Play")
        self.step_button = Button(130, WINDOW_HEIGHT - 60, 100, 40, "Step")
        self.reset_button = Button(240, WINDOW_HEIGHT - 60, 100, 40, "Reset")

        # Fonts
        self.title_font = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)

        # Action log
        self.action_log = []
        self.max_log_entries = 50  # Increased from 20

        # Map view
        self.map_view = MapView(740, 50, 640, 450, self.simulator)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle button clicks
            if self.play_button.handle_event(event):
                self.paused = not self.paused
                self.play_button.text = "Pause" if not self.paused else "Play"

            if self.step_button.handle_event(event):
                self.step_simulation()

            if self.reset_button.handle_event(event):
                self.reset_simulation()

            # Speed control with arrow keys and presets
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.days_per_second = min(100, self.days_per_second * 2)
                elif event.key == pygame.K_DOWN:
                    self.days_per_second = max(0.1, self.days_per_second / 2)
                # Speed presets
                elif event.key == pygame.K_1:
                    self.days_per_second = 0.5
                elif event.key == pygame.K_2:
                    self.days_per_second = 1.0
                elif event.key == pygame.K_3:
                    self.days_per_second = 5.0
                elif event.key == pygame.K_4:
                    self.days_per_second = 10.0
                elif event.key == pygame.K_5:
                    self.days_per_second = 25.0

    def step_simulation(self):
        try:
            # Run all 20 ticks for a complete day
            actions = []
            for _ in range(self.simulator.ticks_per_day):
                day_actions = self.simulator.tick()
                actions.extend(day_actions)

            # Add actions to log
            for action in actions:
                log_entry = f"Day {self.simulator.day}: {action['action']}"
                self.action_log.append(log_entry)
                if self.days_per_second <= 2:  # Only print at slow speeds
                    print(log_entry)

            # Keep only recent entries
            if len(self.action_log) > self.max_log_entries:
                self.action_log = self.action_log[-self.max_log_entries:]

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            self.action_log.append(error_msg)
            print(error_msg)  # Print to stdout
            self.paused = True
            self.play_button.text = "Play"

    def reset_simulation(self):
        self.simulator.reset()
        self.action_log = []
        self.paused = True
        self.play_button.text = "Play"

    def update(self):
        current_time = pygame.time.get_ticks()

        if not self.paused:
            # Calculate elapsed time and accumulate days
            elapsed_ms = current_time - self.last_update_time
            elapsed_sec = elapsed_ms / 1000.0
            self.day_accumulator += elapsed_sec * self.days_per_second

            # Run full days when accumulated
            while self.day_accumulator >= 1.0:
                self.step_simulation()
                self.day_accumulator -= 1.0

        self.last_update_time = current_time

    def draw(self):
        self.screen.fill(WHITE)

        # Title
        title = self.title_font.render("Economy Simulator", True, BLACK)
        self.screen.blit(title, (20, 10))

        # Day counter
        day_text = self.font.render(f"Day: {self.simulator.day}", True, BLACK)
        self.screen.blit(day_text, (WINDOW_WIDTH - 150, 20))

        # Draw person cards
        x_offset = 20
        y_offset = 50
        cards_per_row = 3

        for i, person in enumerate(self.simulator.people):
            row = i // cards_per_row
            col = i % cards_per_row

            x = x_offset + col * 240
            y = y_offset + row * 250

            card = PersonCard(x, y, person)
            card.draw(self.screen)

        # Draw map view
        self.map_view.draw(self.screen)

        # Draw action log
        log_x = 740
        log_y = 510
        log_title = self.font.render("Action Log", True, BLACK)
        self.screen.blit(log_title, (log_x, log_y))

        # Adjusted log area size
        log_width = 640  
        log_height = 210  # Reduced height
        pygame.draw.rect(self.screen, LIGHT_GRAY, (log_x, log_y + 25, log_width, log_height))
        pygame.draw.rect(self.screen, BLACK, (log_x, log_y + 25, log_width, log_height), 2)

        # Draw log entries with smaller font and tighter spacing
        y = log_y + 30
        line_height = 16  
        max_entries = 12  # Show fewer entries due to less space
        for entry in self.action_log[-max_entries:]:  
            if y < log_y + log_height + 15:
                # Truncate long entries based on new width
                if len(entry) > 110:
                    entry = entry[:107] + "..."
                text = self.small_font.render(entry, True, BLACK)
                self.screen.blit(text, (log_x + 10, y))
                y += line_height

        # Draw controls
        self.play_button.draw(self.screen)
        self.step_button.draw(self.screen)
        self.reset_button.draw(self.screen)

        # Speed indicator
        if self.days_per_second >= 1:
            speed_text = self.font.render(f"Speed: {self.days_per_second:.0f} days/sec (↑↓ to adjust)", True, BLACK)
        else:
            speed_text = self.font.render(f"Speed: {self.days_per_second:.1f} days/sec (↑↓ to adjust)", True, BLACK)
        self.screen.blit(speed_text, (360, WINDOW_HEIGHT - 50))

        # Speed preset hint
        preset_text = self.small_font.render("Press 1-5: 0.5x, 1x, 5x, 10x, 25x speed", True, DARK_GRAY)
        self.screen.blit(preset_text, (360, WINDOW_HEIGHT - 30))

        # Update display
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = EconomySimulatorGame()
    game.run()
