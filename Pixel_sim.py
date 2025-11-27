import pygame
import numpy as np
from scipy.ndimage import convolve

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
FPS = 30

class GameOfLife:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Grid: 0 = empty, 1 = Game of Life cell, 2 = Tree
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pixel Simulator with UI Controls")
        self.clock = pygame.time.Clock()
        
        # Pre-allocate working arrays for performance
        self._temp_grid = np.zeros_like(self.grid)
        self._random_array = np.zeros((height, width), dtype=np.float32)
        
        # Zoom and pan variables
        self.zoom_level = 1
        self.max_zoom = 8
        self.min_zoom = 0.5
        self.pan_x = 0
        self.pan_y = 0
        
        # Rules system
        self.rules = {
            'game_of_life': True,
            'trees': False,
        }
        
        # Trees rule parameters
        self.trees_config = {
            'spread_chance': 0.002,
            'spontaneous_chance': 0.00001,
            'spread_radius': 3
        }
        
        # UI system
        self.ui_panel_width = 300
        self.ui_visible = True
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.buttons = []
        self.sliders = []
        
        # Pre-computed kernels for different radii to avoid recreating them
        self.kernels = {}
        self.gol_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        
        # Frame counter for optimization
        self.frame_count = 0
        
        self.setup_ui()
        
    def get_tree_kernel(self, radius):
        """Get or create a cached kernel for tree spreading"""
        if radius not in self.kernels:
            kernel_size = 2 * radius + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            kernel[radius, radius] = 0
            self.kernels[radius] = kernel
        return self.kernels[radius]
        
    def setup_ui(self):
        """Setup UI buttons and sliders"""
        self.buttons = []
        self.sliders = []
        
        y_pos = 20
        
        # Rule toggle buttons
        self.buttons.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos, 200, 30),
            'text': 'Game of Life',
            'action': 'toggle_game_of_life',
            'active': self.rules['game_of_life']
        })
        y_pos += 40
        
        self.buttons.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos, 200, 30),
            'text': 'Trees',
            'action': 'toggle_trees',
            'active': self.rules['trees']
        })
        y_pos += 60
        
        # Trees configuration sliders
        self.sliders.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos + 20, 200, 20),
            'label': 'Spread Chance',
            'value': self.trees_config['spread_chance'],
            'min_val': 0.0,
            'max_val': 0.01,
            'param': 'spread_chance'
        })
        y_pos += 60
        
        self.sliders.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos + 20, 200, 20),
            'label': 'Spontaneous Chance',
            'value': self.trees_config['spontaneous_chance'],
            'min_val': 0.0,
            'max_val': 0.0001,
            'param': 'spontaneous_chance'
        })
        y_pos += 60
        
        self.sliders.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos + 20, 200, 20),
            'label': 'Spread Radius',
            'value': self.trees_config['spread_radius'],
            'min_val': 1,
            'max_val': 10,
            'param': 'spread_radius'
        })
        y_pos += 80
        
        # Control buttons
        self.buttons.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 20, y_pos, 90, 30),
            'text': 'Randomize',
            'action': 'randomize',
            'active': True
        })
        
        self.buttons.append({
            'rect': pygame.Rect(WIDTH - self.ui_panel_width + 120, y_pos, 90, 30),
            'text': 'Clear',
            'action': 'clear',
            'active': True
        })
        y_pos += 40
        
        # UI toggle button
        self.buttons.append({
            'rect': pygame.Rect(WIDTH - 30, 10, 20, 20),
            'text': 'X',
            'action': 'toggle_ui',
            'active': True
        })
        
    def randomize_grid(self, density=0.3):
        """Initialize grid with random live cells"""
        self.grid = np.random.choice([0, 1], size=(self.height, self.width), 
                                   p=[1-density, density]).astype(np.uint8)
    
    def apply_game_of_life_rule(self):
        """Apply Game of Life rules using optimized vectorized operations"""
        if not self.rules['game_of_life']:
            return
        
        # Check if there are any white pixels - optimized check
        has_white_pixels = np.any(self.grid == 1)
        
        # If no white pixels exist, spawn a 2x2 block - optimized version
        if not has_white_pixels:
            # Find empty positions more efficiently
            empty_y, empty_x = np.where(self.grid == 0)
            if len(empty_y) > 0:
                # Filter positions that can fit 2x2 block
                valid_mask = (empty_x < self.width - 1) & (empty_y < self.height - 1)
                valid_y = empty_y[valid_mask]
                valid_x = empty_x[valid_mask]
                
                if len(valid_y) > 0:
                    # Check if 2x2 area is empty for each valid position
                    for i in range(min(100, len(valid_y))):  # Limit checks for performance
                        y, x = valid_y[i], valid_x[i]
                        if (self.grid[y:y+2, x:x+2] == 0).all():
                            self.grid[y:y+2, x:x+2] = 1
                            return
            
        # Optimized Game of Life calculation
        living_grid = (self.grid == 1)
        neighbors = convolve(living_grid.astype(np.uint8), self.gol_kernel, mode='wrap')
        
        gol_survive = living_grid & ((neighbors == 2) | (neighbors == 3))
        gol_born = (self.grid == 0) & (neighbors == 3)
        
        # Tree conversion with 50% chance - optimized
        trees_with_neighbors = (self.grid == 2) & (neighbors > 0)
        if np.any(trees_with_neighbors):
            self._random_array = np.random.random((self.height, self.width))
            trees_convert = trees_with_neighbors & (self._random_array < 0.5)
        else:
            trees_convert = np.zeros_like(self.grid, dtype=bool)
        
        # Combine and apply 5% failure rate
        should_be_white = gol_survive | gol_born | trees_convert
        if np.any(should_be_white):
            self._random_array = np.random.random((self.height, self.width))
            actually_white = should_be_white & (self._random_array >= 0.05)
        else:
            actually_white = np.zeros_like(self.grid, dtype=bool)
        
        # Update grid efficiently
        self._temp_grid.fill(0)
        self._temp_grid[self.grid == 2] = 2  # Preserve trees not being converted
        self._temp_grid[actually_white] = 1
        self.grid, self._temp_grid = self._temp_grid, self.grid
    
    def apply_trees_rule(self):
        """Apply trees spreading rule - highly optimized"""
        if not self.rules['trees']:
            return
        
        empty_cells = (self.grid == 0)
        
        # Optimized spontaneous generation
        if self.trees_config['spontaneous_chance'] > 0 and np.any(empty_cells):
            self._random_array = np.random.random((self.height, self.width))
            spontaneous_trees = empty_cells & (self._random_array < self.trees_config['spontaneous_chance'])
            if np.any(spontaneous_trees):
                self.grid[spontaneous_trees] = 2
                empty_cells = (self.grid == 0)  # Update mask
        
        # Optimized tree spreading
        if self.trees_config['spread_chance'] > 0 and np.any(empty_cells):
            radius = int(self.trees_config['spread_radius'])
            kernel = self.get_tree_kernel(radius)
            
            tree_mask = (self.grid == 2)
            if np.any(tree_mask):
                nearby_trees = convolve(tree_mask.astype(np.uint8), kernel, mode='wrap')
                can_spread = empty_cells & (nearby_trees > 0)
                
                if np.any(can_spread):
                    self._random_array = np.random.random((self.height, self.width))
                    new_trees = can_spread & (self._random_array < self.trees_config['spread_chance'])
                    if np.any(new_trees):
                        self.grid[new_trees] = 2
    
    def update_grid(self):
        """Apply all enabled rules in the correct order"""
        self.frame_count += 1
        
        # Apply trees rule FIRST so trees can grow
        self.apply_trees_rule()
        # Then apply Game of Life rule which can convert trees to white
        self.apply_game_of_life_rule()
    
    def toggle_rule(self, rule_name):
        """Toggle a rule on/off"""
        if rule_name in self.rules:
            self.rules[rule_name] = not self.rules[rule_name]
            # Update button states
            for button in self.buttons:
                if button['action'] == f'toggle_{rule_name}':
                    button['active'] = self.rules[rule_name]
    
    def handle_button_click(self, pos):
        """Handle button clicks"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                if button['action'] == 'toggle_game_of_life':
                    self.toggle_rule('game_of_life')
                elif button['action'] == 'toggle_trees':
                    self.toggle_rule('trees')
                elif button['action'] == 'randomize':
                    self.randomize_grid(density=0.1)
                elif button['action'] == 'clear':
                    self.grid.fill(0)
                elif button['action'] == 'toggle_ui':
                    self.ui_visible = not self.ui_visible
                return True
        return False
    
    def handle_slider_drag(self, pos):
        """Handle slider dragging"""
        for slider in self.sliders:
            if slider['rect'].collidepoint(pos):
                relative_x = pos[0] - slider['rect'].x
                percentage = max(0, min(1, relative_x / slider['rect'].width))
                
                if slider['param'] == 'spread_radius':
                    new_value = int(slider['min_val'] + percentage * (slider['max_val'] - slider['min_val']))
                else:
                    new_value = slider['min_val'] + percentage * (slider['max_val'] - slider['min_val'])
                
                slider['value'] = new_value
                self.trees_config[slider['param']] = new_value
                return True
        return False
    
    def draw_ui(self):
        """Draw the UI panel - optimized to reduce redundant drawing"""
        if not self.ui_visible:
            show_button = pygame.Rect(WIDTH - 30, 10, 20, 20)
            pygame.draw.rect(self.screen, DARK_GRAY, show_button)
            pygame.draw.rect(self.screen, WHITE, show_button, 2)
            text = self.small_font.render('>', True, WHITE)
            self.screen.blit(text, (show_button.x + 6, show_button.y + 2))
            return
        
        # Draw UI panel background
        panel_rect = pygame.Rect(WIDTH - self.ui_panel_width, 0, self.ui_panel_width, HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)
        pygame.draw.line(self.screen, WHITE, (WIDTH - self.ui_panel_width, 0), (WIDTH - self.ui_panel_width, HEIGHT), 2)
        
        # Draw title
        title = self.font.render('Controls', True, WHITE)
        self.screen.blit(title, (WIDTH - self.ui_panel_width + 20, 5))
        
        # Draw buttons
        for button in self.buttons:
            if button['action'] == 'toggle_ui':
                continue
                
            color = GREEN if button['active'] else RED
            if button['action'] in ['randomize', 'clear']:
                color = LIGHT_GRAY
            
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, WHITE, button['rect'], 2)
            
            text = self.font.render(button['text'], True, BLACK if color == LIGHT_GRAY else WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
        
        # Draw sliders
        for slider in self.sliders:
            label_text = self.small_font.render(slider['label'], True, WHITE)
            self.screen.blit(label_text, (slider['rect'].x, slider['rect'].y - 20))
            
            pygame.draw.rect(self.screen, GRAY, slider['rect'])
            pygame.draw.rect(self.screen, WHITE, slider['rect'], 2)
            
            percentage = (slider['value'] - slider['min_val']) / (slider['max_val'] - slider['min_val'])
            handle_x = slider['rect'].x + percentage * slider['rect'].width
            handle_rect = pygame.Rect(handle_x - 5, slider['rect'].y - 2, 10, slider['rect'].height + 4)
            pygame.draw.rect(self.screen, WHITE, handle_rect)
            
            if slider['param'] == 'spread_radius':
                value_text = f"{int(slider['value'])}"
            elif slider['param'] == 'spontaneous_chance':
                value_text = f"{slider['value']:.6f}"
            else:
                value_text = f"{slider['value']:.4f}"
            value_surface = self.small_font.render(value_text, True, WHITE)
            self.screen.blit(value_surface, (slider['rect'].x + slider['rect'].width + 10, slider['rect'].y))
        
        # Draw hide UI button
        hide_button = self.buttons[-1]
        pygame.draw.rect(self.screen, RED, hide_button['rect'])
        pygame.draw.rect(self.screen, WHITE, hide_button['rect'], 2)
        text = self.small_font.render(hide_button['text'], True, WHITE)
        text_rect = text.get_rect(center=hide_button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def screen_to_grid(self, screen_x, screen_y):
        """Convert screen coordinates to grid coordinates"""
        return (int((screen_x / self.zoom_level) + self.pan_x),
                int((screen_y / self.zoom_level) + self.pan_y))
    
    def draw(self):
        """Draw the grid to screen - heavily optimized rendering"""
        self.screen.fill(BLACK)
        
        sim_width = WIDTH - (self.ui_panel_width if self.ui_visible else 0)
        
        # Calculate visible area with bounds checking
        start_x = max(0, int(self.pan_x))
        start_y = max(0, int(self.pan_y))
        end_x = min(self.width, int(self.pan_x + sim_width / self.zoom_level) + 1)
        end_y = min(self.height, int(self.pan_y + HEIGHT / self.zoom_level) + 1)
        
        visible_grid = self.grid[start_y:end_y, start_x:end_x]
        
        if visible_grid.size > 0:
            # Create RGB array more efficiently
            h, w = visible_grid.shape
            rgb_array = np.zeros((w, h, 3), dtype=np.uint8)
            
            # Use advanced indexing for faster color assignment
            white_mask = visible_grid.T == 1
            green_mask = visible_grid.T == 2
            
            rgb_array[white_mask] = WHITE
            rgb_array[green_mask] = GREEN
            
            if abs(self.zoom_level - 1.0) < 0.01:  # Approximately 1x zoom
                temp_surface = pygame.Surface((w, h))
                pygame.surfarray.blit_array(temp_surface, rgb_array)
                self.screen.blit(temp_surface, 
                               (int((start_x - self.pan_x) * self.zoom_level),
                                int((start_y - self.pan_y) * self.zoom_level)))
            else:
                if w > 0 and h > 0:
                    temp_surface = pygame.Surface((w, h))
                    pygame.surfarray.blit_array(temp_surface, rgb_array)
                    
                    scaled_width = max(1, int(w * self.zoom_level))
                    scaled_height = max(1, int(h * self.zoom_level))
                    
                    scaled_surface = pygame.transform.scale(temp_surface, 
                                                           (scaled_width, scaled_height))
                    self.screen.blit(scaled_surface, 
                                   (int((start_x - self.pan_x) * self.zoom_level),
                                    int((start_y - self.pan_y) * self.zoom_level)))
        
        self.draw_ui()
        pygame.display.flip()
    
    def handle_mouse(self):
        """Optimized mouse handling"""
        mouse_buttons = pygame.mouse.get_pressed()
        if not any(mouse_buttons):
            return
            
        mouse_x, mouse_y = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()
        
        if self.ui_visible and mouse_x > WIDTH - self.ui_panel_width:
            return
        
        grid_x, grid_y = self.screen_to_grid(mouse_x, mouse_y)
        brush_size = max(1, int(3 / self.zoom_level))
        
        if mouse_buttons[0]:  # Left mouse button - draw
            cell_type = 2 if keys[pygame.K_LSHIFT] else 1
            
            x_start = max(0, grid_x - brush_size//2)
            x_end = min(self.width, grid_x + brush_size//2 + 1)
            y_start = max(0, grid_y - brush_size//2)
            y_end = min(self.height, grid_y + brush_size//2 + 1)
            
            self.grid[y_start:y_end, x_start:x_end] = cell_type
        
        elif mouse_buttons[2]:  # Right mouse button - erase
            x_start = max(0, grid_x - brush_size//2)
            x_end = min(self.width, grid_x + brush_size//2 + 1)
            y_start = max(0, grid_y - brush_size//2)
            y_end = min(self.height, grid_y + brush_size//2 + 1)
            
            self.grid[y_start:y_end, x_start:x_end] = 0
    
    def handle_zoom(self, event):
        """Handle zoom events"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        if self.ui_visible and mouse_x > WIDTH - self.ui_panel_width:
            return
        
        old_grid_x, old_grid_y = self.screen_to_grid(mouse_x, mouse_y)
        
        if event.button == 4:  # Scroll up - zoom in
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
        elif event.button == 5:  # Scroll down - zoom out
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
        
        new_grid_x, new_grid_y = self.screen_to_grid(mouse_x, mouse_y)
        self.pan_x += old_grid_x - new_grid_x
        self.pan_y += old_grid_y - new_grid_y
        
        sim_width = WIDTH - (self.ui_panel_width if self.ui_visible else 0)
        max_pan_x = max(0, self.width - sim_width / self.zoom_level)
        max_pan_y = max(0, self.height - HEIGHT / self.zoom_level)
        self.pan_x = max(0, min(max_pan_x, self.pan_x))
        self.pan_y = max(0, min(max_pan_y, self.pan_y))
    
    def run(self):
        """Main game loop - optimized"""
        running = True
        paused = False
        panning = False
        dragging_slider = False
        last_mouse_pos = (0, 0)
        
        self.randomize_grid(density=0.1)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_HOME:
                        self.zoom_level = 1
                        self.pan_x = 0
                        self.pan_y = 0
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if not self.handle_button_click(event.pos):
                            if not self.handle_slider_drag(event.pos):
                                if not self.ui_visible or event.pos[0] < WIDTH - self.ui_panel_width:
                                    pass
                            else:
                                dragging_slider = True
                    elif event.button == 2:
                        if not self.ui_visible or event.pos[0] < WIDTH - self.ui_panel_width:
                            panning = True
                            last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button in [4, 5]:
                        self.handle_zoom(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging_slider = False
                    elif event.button == 2:
                        panning = False
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_slider:
                        self.handle_slider_drag(event.pos)
                    elif panning:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        dx = (mouse_x - last_mouse_pos[0]) / self.zoom_level
                        dy = (mouse_y - last_mouse_pos[1]) / self.zoom_level
                        
                        self.pan_x -= dx
                        self.pan_y -= dy
                        
                        sim_width = WIDTH - (self.ui_panel_width if self.ui_visible else 0)
                        max_pan_x = max(0, self.width - sim_width / self.zoom_level)
                        max_pan_y = max(0, self.height - HEIGHT / self.zoom_level)
                        self.pan_x = max(0, min(max_pan_x, self.pan_x))
                        self.pan_y = max(0, min(max_pan_y, self.pan_y))
                        
                        last_mouse_pos = (mouse_x, mouse_y)
            
            # Only show UI toggle if UI is hidden
            if not self.ui_visible and pygame.mouse.get_pressed()[0]:
                show_button = pygame.Rect(WIDTH - 30, 10, 20, 20)
                if show_button.collidepoint(pygame.mouse.get_pos()):
                    self.ui_visible = True
            
            self.handle_mouse()
            
            if not paused:
                self.update_grid()
            
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = GameOfLife(WIDTH, HEIGHT)
    print("Controls:")
    print("SPACE - Pause/Unpause")
    print("ESC - Quit")
    print("HOME - Reset zoom and pan")
    print("Left Mouse - Draw Game of Life cells")
    print("SHIFT + Left Mouse - Draw trees")
    print("Right Mouse - Erase cells")
    print("Middle Mouse + Drag - Pan around")
    print("Mouse Wheel - Zoom in/out")
    print("\nUse the UI panel on the right to control rules and settings!")
    game.run()