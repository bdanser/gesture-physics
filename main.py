import cv2 as cv
from camera.camera import get_frame, release
from render.renderer import draw_ball, draw_landmarks, draw_text, draw_slider
from gestures.gesture_model import get_landmarks
from physics.ball import Ball
from physics.physics_engine import is_grabbing, update
from utils.config import SCREEN_HEIGHT, SCREEN_WIDTH
import torch
import pickle
from gestures.gesture_model import GestureModel
from utils.math_utils import distance

# create ball at center of screen with initial velocity
balls = [Ball(960, 540)]
balls[0].vx = 5
balls[0].vy = -10

holding_hand_label = None
prev_palm_pos = {}
hand_vel = {}
vel_history = {}

gravity = 20

model = GestureModel()
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

peace_cooldown = 0

with open('gesture_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

def predict_gesture(hand):
    landmarks_flat = []
    for lm in hand:
        landmarks_flat.extend([lm.x, lm.y])
    input_tensor = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return encoder.inverse_transform([prediction])[0]

def get_closest_ball(balls, x, y):
    return min(balls, key=lambda b: distance((b.x, b.y), (x, y)))

while True:
    # get latest webcam frame
    frame = get_frame()
    if frame is None:
        continue

    # mirror the frame
    frame = cv.flip(frame, 1)

    # convert to RGB for MediaPipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # detect hand landmarks
    results = get_landmarks(frame_rgb)
    draw_landmarks(frame, results.hand_landmarks)

    if results.hand_landmarks:
        # get first detected hand
        hand = results.hand_landmarks[0]
        label = results.handedness[0][0].display_name

        # run hand through PyTorch model to get gesture name
        gesture = predict_gesture(hand)

        # draw gesture label above wrist
        wrist = hand[0]
        wx = int(wrist.x * frame.shape[1])
        wy = int(wrist.y * frame.shape[0])
        draw_text(frame, gesture, (wx, wy - 20))

        # get palm position
        palm = hand[9]
        px = palm.x * frame.shape[1]
        py = palm.y * frame.shape[0]

        ball = get_closest_ball(balls, px, py)

        # calculate hand velocity by comparing current and previous palm position
        prev = prev_palm_pos.get(label)
        if prev:
            vx = px - prev[0]
            vy = py - prev[1]

            # keep a rolling history of the last 5 velocity samples
            if label not in vel_history:
                vel_history[label] = []
            vel_history[label].append((vx, vy))
            if len(vel_history[label]) > 5:
                vel_history[label].pop(0)

            # average the velocity history to smooth out spikes
            hand_vel[label] = (
                sum(v[0] for v in vel_history[label]) / len(vel_history[label]),
                sum(v[1] for v in vel_history[label]) / len(vel_history[label])
            )
        else:
            hand_vel[label] = (0, 0)

        # update previous palm position for next frame
        prev_palm_pos[label] = (px, py)

        # grab the ball if fist detected and hand is close enough
        if not ball.held and gesture == 'fist' and is_grabbing(hand, ball, SCREEN_WIDTH, SCREEN_HEIGHT):
            ball.held = True
            holding_hand_label = label

        # move ball with hand while held, release with throw velocity when fist opens
        if ball.held and label == holding_hand_label:
            if gesture == 'fist':
                ball.x = px
                ball.y = py
            else:
                ball.held = False
                ball.vx = hand_vel[label][0]
                ball.vy = hand_vel[label][1]
                holding_hand_label = None

        # point gesture — push ball away from finger
        if gesture == 'point':
            index_tip = hand[8]
            tip_x = index_tip.x * frame.shape[1]
            tip_y = index_tip.y * frame.shape[0]
            dist = distance((tip_x, tip_y), (ball.x, ball.y))

            if dist < ball.radius + 20 and dist > 0:
                dx = ball.x - tip_x
                dy = ball.y - tip_y
                # normalize direction
                ball.vx += (dx / dist) * 15
                ball.vy += (dy / dist) * 15
                # position correction
                ball.x = tip_x + (dx / dist) * (ball.radius + 20)
                ball.y = tip_y + (dy / dist) * (ball.radius + 20)
                ball.held = False

        if gesture == 'peace' and len(balls) < 5 and peace_cooldown == 0:
            new_ball = Ball(int(px), int(py))
            new_ball.vx = 5
            new_ball.vy = -10
            balls.append(new_ball)
            peace_cooldown = 30

        if peace_cooldown > 0:
            peace_cooldown -= 1

        # gravity slider — index finger near right edge controls gravity
        index_tip = hand[8]
        tip_x = index_tip.x * frame.shape[1]
        tip_y = index_tip.y * frame.shape[0]
        if abs(tip_x - 1870) < 100:
            t = 1 - (tip_y - 100) / (500 - 100)
            t = max(0, min(1, t))
            gravity = t * 80

    for ball in balls:
        if not ball.held:
            update(ball, gravity)

        draw_ball(frame, ball)

    draw_slider(frame, gravity)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

release()
