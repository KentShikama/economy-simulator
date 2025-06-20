import pygame
import sys
from simulator import EconomySimulator

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS = 60

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
        
        # City and Money
        city_text = f"City: {self.person.city}"
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


class CityView:
    def __init__(self, x, y, city_name):
        self.x = x
        self.y = y
        self.city_name = city_name
        self.width = 180
        self.height = 120
        self.font = pygame.font.Font(None, 20)
        
    def draw(self, screen, people_in_city):
        # Background
        pygame.draw.rect(screen, LIGHT_GRAY, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        
        # City name
        title = self.font.render(f"City {self.city_name}", True, BLACK)
        screen.blit(title, (self.x + self.width//2 - title.get_width()//2, self.y + 10))
        
        # Draw people icons
        y_offset = self.y + 40
        for i, person in enumerate(people_in_city):
            if i >= 5:  # Max 5 people shown
                break
            
            type_name = type(person).__name__
            color = TYPE_COLORS.get(type_name, BLACK)
            
            # Person dot
            pygame.draw.circle(screen, color, (self.x + 20, y_offset), 8)
            
            # Person name
            font_small = pygame.font.Font(None, 16)
            name_text = font_small.render(person.name, True, BLACK)
            screen.blit(name_text, (self.x + 35, y_offset - 8))
            
            y_offset += 20


class EconomySimulatorGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Economy Simulator")
        self.clock = pygame.time.Clock()
        
        self.simulator = EconomySimulator()
        self.running = True
        self.paused = True
        self.simulation_speed = 1000  # milliseconds between steps
        self.last_step_time = 0
        
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
        self.max_log_entries = 20
        
        # City views
        self.city_views = {
            'A': CityView(960, 50, 'A'),
            'B': CityView(960, 180, 'B'),
            'C': CityView(960, 310, 'C')
        }
        
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
            
            # Speed control with arrow keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.simulation_speed = min(5000, self.simulation_speed + 100)
                elif event.key == pygame.K_RIGHT:
                    self.simulation_speed = max(100, self.simulation_speed - 100)
    
    def step_simulation(self):
        try:
            actions = self.simulator.simulate_day()
            
            # Add actions to log
            for action in actions:
                log_entry = f"Day {self.simulator.day}: {action['action']}"
                self.action_log.append(log_entry)
            
            # Keep only recent entries
            if len(self.action_log) > self.max_log_entries:
                self.action_log = self.action_log[-self.max_log_entries:]
                
        except Exception as e:
            self.action_log.append(f"ERROR: {str(e)}")
            self.paused = True
            self.play_button.text = "Play"
    
    def reset_simulation(self):
        self.simulator.reset()
        self.action_log = []
        self.paused = True
        self.play_button.text = "Play"
    
    def update(self):
        current_time = pygame.time.get_ticks()
        
        if not self.paused and current_time - self.last_step_time > self.simulation_speed:
            self.step_simulation()
            self.last_step_time = current_time
    
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
        
        # Draw city views
        for city_name, city_view in self.city_views.items():
            people_in_city = [p for p in self.simulator.people if p.city == city_name]
            city_view.draw(self.screen, people_in_city)
        
        # Draw action log
        log_x = 960
        log_y = 440
        log_title = self.font.render("Action Log", True, BLACK)
        self.screen.blit(log_title, (log_x, log_y))
        
        pygame.draw.rect(self.screen, LIGHT_GRAY, (log_x, log_y + 25, 420, 280))
        pygame.draw.rect(self.screen, BLACK, (log_x, log_y + 25, 420, 280), 2)
        
        # Draw log entries
        y = log_y + 35
        for entry in self.action_log[-12:]:  # Show last 12 entries
            if y < log_y + 295:
                # Truncate long entries
                if len(entry) > 52:
                    entry = entry[:49] + "..."
                text = self.small_font.render(entry, True, BLACK)
                self.screen.blit(text, (log_x + 10, y))
                y += 20
        
        # Draw controls
        self.play_button.draw(self.screen)
        self.step_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        
        # Speed indicator
        speed_text = self.font.render(f"Speed: {self.simulation_speed/1000:.1f}s (use ← →)", True, BLACK)
        self.screen.blit(speed_text, (360, WINDOW_HEIGHT - 50))
        
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