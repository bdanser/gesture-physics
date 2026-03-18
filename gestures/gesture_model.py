import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch.nn as nn
import torch

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def get_landmarks(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(mp_image)
    return results

class GestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(42, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
