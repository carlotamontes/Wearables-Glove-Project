import pygame
import sys

# Initialize pygame
pygame.init()

# Window
WIDTH = 900
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Circle Game")

# Clock
clock = pygame.time.Clock()
FPS = 60

# Colors
BACKGROUND = (245, 234, 218)
DARK = (20, 35, 50)
YELLOW = (245, 200, 0)

# Wire
wire_y = HEIGHT // 2

# Ring / circle (start above the wire)
circle_x = WIDTH // 2
initial_circle_y = wire_y - 150
circle_y = initial_circle_y
circle_radius = 70
circle_thickness = 16

# Physics
velocity_y = 0
gravity = 0.7
jump_strength = -13

# Game state
started = False
game_over = False

# Font
font = pygame.font.SysFont(None, 48)

def draw_text(text, y, color=DARK):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(WIDTH // 2, y))
    screen.blit(surf, rect)

def reset():
    global circle_y, velocity_y, started, game_over
    circle_y = initial_circle_y
    velocity_y = 0
    started = False
    game_over = False

reset()

while True:
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Press SPACE to start/jump/restart
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not started:
                    started = True
                    game_over = False
                    velocity_y = jump_strength
                elif game_over:
                    reset()
                    started = True
                    velocity_y = jump_strength
                else:
                    velocity_y = jump_strength

    # Physics update (only while started and not game over)
    if started and not game_over:
        velocity_y += gravity
        circle_y += velocity_y

        # Collision detection between the wire (horizontal line) and the ring border
        d = abs(circle_y - wire_y)
        inner_border = circle_radius - circle_thickness
        # if the horizontal line crosses the ring's drawn border region -> collision
        if inner_border <= d <= circle_radius:
            game_over = True

        # Limit movement so it does not fall forever
        if circle_y > wire_y:
            circle_y = wire_y
            velocity_y = 0

    # Draw background
    screen.fill(BACKGROUND)

    # Draw wire
    pygame.draw.line(screen, DARK, (0, wire_y), (WIDTH, wire_y), 10)

    # Draw ring (drawn after wire so the wire shows through the hollow center)
    pygame.draw.circle(screen, YELLOW, (circle_x, int(circle_y)), circle_radius, circle_thickness)

    # HUD / messages
    if not started and not game_over:
        draw_text("Press SPACE to start", HEIGHT // 4)
    if game_over:
        draw_text("Game Over - Press SPACE to restart", HEIGHT // 4, color=(200, 20, 20))

    # Update display
    pygame.display.flip()
    clock.tick(FPS)