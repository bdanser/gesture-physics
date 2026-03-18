# gesture-physics

Real-time AR ball interaction system using MediaPipe hand tracking and a PyTorch gesture classifier. Grab, throw, and interact with physics-based virtual balls through natural hand gestures captured via webcam.

## Features

- Real-time hand tracking using MediaPipe
- PyTorch neural network gesture classifier trained on custom data
- Physics simulation with gravity, velocity, and wall bouncing
- Grab and throw balls with a fist gesture
- Push balls with a point gesture
- Spawn new balls with a peace sign gesture
- AR gravity slider controlled by your index finger
- Multi-ball support

## Requirements

- Python 3.12
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```
git clone https://github.com/bdanser/gesture-physics.git
cd gesture-physics
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install opencv-python mediapipe torch scikit-learn pandas numpy
```

4. Download the MediaPipe hand landmark model:
```
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

5. Download the pre-trained gesture model files from the releases page and place them in the project root:
- `gesture_model.pth`
- `gesture_encoder.pkl`

## How to use

Run the main script:
```
python main.py
```

| Gesture | Action |
|---|---|
| Fist near ball | Grab the ball |
| Open hand | Release and throw |
| Point finger | Push the ball |
| Peace sign | Spawn a new ball |
| Index finger on right edge | Control gravity slider |

Press `q` to quit.

## Known limitations

- Optimized for single-hand interaction. Two-hand support is limited by MediaPipe's landmark accuracy on consumer webcams at close range.
- Best results with good lighting and a contrasting background.
- Gesture classifier performance depends on the quality and variety of training data collected.

## Project structure
```
gesture-physics/
    main.py
    camera/
        camera.py
    vision/
        hand_tracker.py
    gestures/
        gesture_logic.py
        gesture_model.py
        collect_data.py
        train_model.py
    physics/
        ball.py
        physics_engine.py
    render/
        renderer.py
    utils/
        math_utils.py
        config.py
```
