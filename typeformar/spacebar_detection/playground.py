import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

FEATURE_DIM = 1  # 21 * 3 # 21 landmarks * 3 coordinates (x, y, z)
HIDDEN_DIM = 5  # hyperparameter to be tuned
OUTPUT_DIM = 2  # nothing, spacebar down, or spacebar up

EPOCHS = 100

########################################################
# Architecture
########################################################

output_to_ix = {"not_pressed": 0, "pressed": 1}


class SpacebarDetectorLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            feature_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # Linear layer to map from hidden state to output (2 * hidden_dim because bidirectional)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, landmarks_sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(landmarks_sequence)

        out_space = self.fc(lstm_out.view(len(landmarks_sequence), -1))
        out_scores = F.softmax(out_space, dim=1)

        return out_scores


########################################################
# Prepare Training Data
########################################################

# fmt: off
training_data = [
    {
        "landmarks": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,],
        "output": 10 * ["not_pressed"] + 7 * ["pressed"] + 10 * ["not_pressed"],
    },
    {
        "landmarks": [10, 5, 2, 0, -2, 0, 5, 20],
        "output": 3 * ["not_pressed"] + ["pressed"] + ["not_pressed"] + ["pressed"] + 2 * ["not_pressed"],
    },
    {
        "landmarks": [1, 1, 0, 0, 0, 0, 1, 1, 1],
        "output": 2 * ["not_pressed"] + 4 * ["pressed"] + 3 * ["not_pressed"],
    },
    {
        "landmarks": [2, 1, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 3, 4, 5],
        "output": 2 * ["not_pressed"] + 3 * ["pressed"] + 6 * ["not_pressed"] + 3 * ["pressed"] + 3 * ["not_pressed"],
    },
    {
        "landmarks": [1, 0],
        "output": ["not_pressed", "pressed"],
    },
    {
        "landmarks": [0, 1],
        "output": ["not_pressed", "pressed"],
    },
    {
        "landmarks": [0, 1, 2, 0],
        "output": ["pressed", "not_pressed", "not_pressed", "pressed"],
    },
    {
        "landmarks": [56, 54, 0, 1, 2, 0],
        "output": ["not_pressed", "not_pressed", "pressed", "not_pressed", "not_pressed", "pressed"],
    },
    {
        "landmarks": [56, 54, 0, 50, 50, 0],
        "output": ["not_pressed", "not_pressed", "pressed", "not_pressed", "not_pressed", "pressed"],
    },
    {
        "landmarks": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        "output": ["not_pressed", "pressed", "pressed", "not_pressed", "pressed", "pressed", "not_pressed", "pressed", "pressed", "not_pressed", "pressed", "pressed", "not_pressed"],
    },
    {
        "landmarks": [5, 0, 0, 0, 0, 0, 0, 5],
        "output": ["not_pressed"] + 6 * ["pressed"] + ["not_pressed"],
    }
]

def prepare_input(landmarks):
    return torch.tensor(landmarks, dtype=torch.float32).view(len(landmarks), FEATURE_DIM)

def prepare_output(output):
    return torch.tensor([output_to_ix[o] for o in output], dtype=torch.long)

def print_output_prediction(output):
    # Print the output predictions for each landmark
    pred = torch.argmax(output, dim=1)
    print(" ".join(["_" if o == 0 else "P" for o in pred]))



########################################################
# Training
########################################################

model = SpacebarDetectorLSTM(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# See what the scores are before training
with torch.no_grad():
    landmarks = training_data[-1]["landmarks"]
    inputs = prepare_input(landmarks)
    out_scores = model(inputs)
    print_output_prediction(out_scores)

# Train the model
for epoch in range(EPOCHS):
    for datum in training_data:
        landmarks = datum["landmarks"]
        outputs = datum["output"]

        # Clear the gradients
        model.zero_grad()

        # Prepare the input and output
        inputs = prepare_input(landmarks)
        outputs = prepare_output(outputs)

        # Forward pass
        out_scores = model(inputs)

        # Update parameters
        loss = loss_fn(out_scores, outputs)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")

# See what the scores are after training
with torch.no_grad():
    landmarks = [500, 30, 0, 0, 30, 40, 0, -1, 50]
    inputs = prepare_input(landmarks)
    out_scores = model(inputs)
    print_output_prediction(out_scores)
