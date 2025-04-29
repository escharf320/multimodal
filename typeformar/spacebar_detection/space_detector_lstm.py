import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typeformar.spacebar_detection.dataset_preparation import prepare_dataset

torch.manual_seed(1)

FEATURE_DIM = 624  # 20 * 3 * 2 * 2  # 20 landmarks * 3 coordinates (x, y, z)
HIDDEN_DIM = 20  # hyperparameter to be tuned
OUTPUT_DIM = 2  # nothing, spacebar down, or spacebar up

EPOCHS = 600

########################################################
# Architecture
########################################################

output_to_ix = {"not_pressed": 0, "pressed": 1}


class SpacebarDetectorLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            bidirectional=True,
            num_layers=5,
        )

        # Linear layer to map from hidden state to output (2 * hidden_dim because bidirectional)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, landmarks_sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(landmarks_sequence)

        # only take the middle element of the sequence
        out_space = self.fc(lstm_out[len(landmarks_sequence) // 2].view(1, -1))
        out_scores = F.softmax(out_space, dim=1)

        return out_scores


########################################################
# Prepare Training Data
########################################################

dataset = prepare_dataset()

print("Dataset size: ", len(dataset))


def prepare_input(landmarks):
    return torch.tensor(landmarks, dtype=torch.float32).view(
        len(landmarks), FEATURE_DIM
    )


def prepare_output(output):
    return torch.tensor([output_to_ix[o] for o in output], dtype=torch.long)


def print_output_prediction(pred):
    # Print the output predictions for each landmark
    print(" ".join(["_" if o == 0 else "P" for o in pred]))


########################################################
# Training
########################################################

model = SpacebarDetectorLSTM(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# See what the scores are before training
# with torch.no_grad():
#     feature_sequence = dataset[-1][0]
#     out_scores = model(feature_sequence)
#     print_output_prediction(out_scores)

# Train the model
for epoch in range(EPOCHS):
    for feature_sequence, output_sequence in dataset:
        # Clear the gradients
        model.zero_grad()

        # Forward pass
        out_scores = model(feature_sequence)

        # Update parameters
        loss = loss_fn(out_scores, output_sequence)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")

# See what the scores are after training
with torch.no_grad():
    first_test_sequence = dataset[0]
    feature_sequence, ground_truth = first_test_sequence
    out_scores = model(feature_sequence)
    print("Prediction:")
    print_output_prediction(torch.argmax(out_scores, dim=1))
    print("Ground truth:")
    print_output_prediction(ground_truth)
    print("Accuracy:")
    print(
        torch.sum(torch.argmax(out_scores, dim=1) == ground_truth) / len(ground_truth)
    )

# Save the model to models directory
models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "spacebar_detector_lstm.pth")
torch.save(model.state_dict(), model_path)
