from utils.math_utils import distance
from utils.config import GRAB_THRESHOLD, SCREEN_WIDTH, SCREEN_HEIGHT

def update(ball, gravity):
    if ball.held:
        return

    ball.vy += gravity
    ball.x += ball.vx
    ball.y += ball.vy

    # Bounce off ceiling
    if ball.y - ball.radius < 0:
        ball.y = ball.radius
        ball.vy *= -0.9
        ball.vx *= 0.98

    # bounce off left wall
    if ball.x - ball.radius < 0:
        ball.x = ball.radius
        ball.vx *= -0.9
        ball.vy *= 0.98

    # bounce off right wall
    if ball.x + ball.radius > SCREEN_WIDTH:
        ball.x = SCREEN_WIDTH - ball.radius
        ball.vx *= -0.9
        ball.vy *= 0.98

    # bounce off floor
    if ball.y + ball.radius > SCREEN_HEIGHT:
        ball.y = SCREEN_HEIGHT - ball.radius
        ball.vy *= -0.9
        ball.vx *= 0.98

def is_grabbing(hand, ball, frame_width, frame_height):

    palm = hand[9]
    palm_x = palm.x * frame_width
    palm_y = palm.y * frame_height

    d = distance((palm_x, palm_y), (ball.x, ball.y))

    return d < GRAB_THRESHOLD
