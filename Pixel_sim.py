import pygame
import numpy as np
from scipy.ndimage import convolve

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1024, 1024
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FPS = 30  # Increased FPS since it will run faster

class GameOfLife:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Game of Life - 1024x1024 (Optimized)")
        self.clock = pygame.time.Clock()
        
        # Convolution kernel for counting neighbors
        self.kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
        
    def randomize_grid(self, density=0.3):
        """Initialize grid with random live cells"""
        self.grid = np.random.choice([0, 1], size=(self.height, self.width), p=[1-density, density]).astype(np.uint8)
    
    def update_grid(self):
        """Apply Game of Life rules using vectorized operations"""
        # Count neighbors using convolution
        neighbors = convolve(self.grid, self.kernel, mode='wrap')
        
        # Apply Game of Life rules vectorized
        # Alive cells with 2-3 neighbors survive, dead cells with 3 neighbors are born
        self.grid = ((self.grid == 1) & ((neighbors == 2) | (neighbors == 3)) | 
                    (self.grid == 0) & (neighbors == 3)).astype(np.uint8)
    
    def draw(self):
        """Draw the grid to screen using optimized surface operations"""
        # Create RGB array directly
        rgb_array = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        rgb_array[self.grid.T == 1] = WHITE  # Transpose for pygame coordinate system
        
        # Use pygame.surfarray for fast blitting
        pygame.surfarray.blit_array(self.screen, rgb_array)
        pygame.display.flip()
    
    def handle_mouse(self):
        """Allow drawing with mouse - now supports brush size"""
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            mouse_x, mouse_y = pygame.mouse.get_pos()
            brush_size = 3  # 3x3 brush for easier drawing
            
            for dx in range(-brush_size//2, brush_size//2 + 1):
                for dy in range(-brush_size//2, brush_size//2 + 1):
                    nx, ny = mouse_x + dx, mouse_y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.grid[ny, nx] = 1
    
    def run(self):
        """Main game loop"""
        running = True
        paused = False
        
        # Initialize with random pattern
        self.randomize_grid(density=0.2)  # Lower density for better performance
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.randomize_grid()
                    elif event.key == pygame.K_c:
                        self.grid.fill(0)
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Handle mouse drawing
            self.handle_mouse()
            
            # Update simulation
            if not paused:
                self.update_grid()
            
            # Draw everything
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = GameOfLife(WIDTH, HEIGHT)
    print("Controls:")
    print("SPACE - Pause/Unpause")
    print("R - Randomize grid")
    print("C - Clear grid")
    print("ESC - Quit")
    print("Left Mouse - Draw cells (3x3 brush)")
    game.run()