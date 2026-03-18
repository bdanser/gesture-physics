import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Data to use for training
train_data = pd.read_csv('gesture_data.csv', header=None)

# Landmark Coordinates
X = train_data.iloc[:, :-1].values

# Gesture labels
y = train_data.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

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

model = GestureModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    # Show accuracy every 10 iterations
    # if epoch % 10 == 0:
    #     print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# with torch.no_grad():
#     outputs = model(X_tensor)
#     predictions = torch.argmax(outputs, dim=1)
#     accuracy = (predictions == y_tensor).float().mean()
#     print(f'Training accuracy: {accuracy:.2%}')

torch.save(model.state_dict(), 'gesture_model.pth')

with open('gesture_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print('Model saved!')
