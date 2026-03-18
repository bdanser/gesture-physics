import cv2 as cv

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Draw the ball on the screen
def draw_ball(frame, ball):
    cv.circle(frame, (int(ball.x), int(ball.y)), ball.radius, (255, 0, 0), -1)

    return frame

# Draw the landmarks on hand as well as connections between them (just preference)
def draw_landmarks(frame, landmarks):
    for hand in landmarks:
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
        for a, b in HAND_CONNECTIONS:
            x1 = int(hand[a].x * frame.shape[1])
            y1 = int(hand[a].y * frame.shape[0])
            x2 = int(hand[b].x * frame.shape[1])
            y2 = int(hand[b].y * frame.shape[0])
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return frame

# When making a fist, the text 'fist' should appear next to that hand
def draw_text(frame, text, position=(50, 50)):
    cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def draw_slider(frame, gravity, min_val=0, max_val=80):
    # slider track position
    track_x = 1870
    track_top = 100
    track_bottom = 500

    # draw the track
    cv.line(frame, (track_x, track_top), (track_x, track_bottom), (200, 200, 200), 2)

    t =  1 - (gravity - min_val) / (max_val - min_val)
    handle_y = int(track_top + t * (track_bottom - track_top))

    cv.circle(frame, (track_x, handle_y), 15, (0, 0, 0), -1)

    cv.putText(frame, f'gravity: {int(gravity)}', (track_x - 120, handle_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
