import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

FEATURE_DIM = 1  # 20 * 3 * 2 * 2  # 20 landmarks * 3 coordinates (x, y, z)
HIDDEN_DIM = 5  # hyperparameter to be tuned
OUTPUT_DIM = 2  # nothing, spacebar down, or spacebar up

EPOCHS = 300

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
            num_layers=18,
            batch_first=True,
        )

        # Linear layer to map from hidden state to output (2 * hidden_dim because bidirectional)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, landmarks_sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(landmarks_sequence)

        # only take the last element of the sequence
        out_space = self.fc(lstm_out[-1].view(1, -1))
        out_scores = F.softmax(out_space, dim=1)

        return out_scores


########################################################
# Prepare Training Data
########################################################

dataset = [
    (
        torch.tensor([[0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).T,
        torch.tensor([0]),
    ),
    (
        torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32).T,
        torch.tensor([0]),
    ),
    (
        torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=torch.float32).T,
        torch.tensor([1]),
    ),
    (
        torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32).T,
        torch.tensor([1]),
    ),
    (
        torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32).T,
        torch.tensor([1]),
    ),
]

test_dataset = [
    (
        torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32).T,
        torch.tensor([1]),
    ),
    (
        torch.tensor([[1, 1, 1, 0, 1, 1, 1]], dtype=torch.float32).T,
        torch.tensor([0]),
    ),
]

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

loss_fn_weights = torch.zeros([2])

# Calculate class weights for the loss function
# Count the number of positive and negative examples in the dataset
positive_count = 0
negative_count = 0
total_samples = 0

for _, output_sequence in dataset:
    for label in output_sequence:
        if label == 1:  # Spacebar pressed
            positive_count += 1
        else:  # Spacebar not pressed
            negative_count += 1
        total_samples += 1

# Calculate weights inversely proportional to class frequencies
# This helps address class imbalance by giving more weight to the minority class
if positive_count > 0 and negative_count > 0:
    weight_negative = total_samples / (2 * negative_count)
    weight_positive = total_samples / (2 * positive_count)
    loss_fn_weights = torch.tensor([weight_negative, weight_positive])
    print(
        f"Class weights: Not pressed = {weight_negative:.4f}, Pressed = {weight_positive:.4f}"
    )
else:
    print("Warning: One of the classes has zero samples. Using equal weights.")
    loss_fn_weights = torch.tensor([1.0, 1.0])


model = SpacebarDetectorLSTM(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_fn = nn.CrossEntropyLoss(weight=loss_fn_weights)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# See what the scores are before training
with torch.no_grad():
    feature_sequence, output_sequence = dataset[-1]
    print(feature_sequence)
    print_output_prediction(output_sequence)
    out_scores = model(feature_sequence)
    print_output_prediction(torch.argmax(out_scores, dim=1))

# Train the model
for epoch in range(EPOCHS):
    for feature_sequence, output_sequence in dataset[:10]:
        # Clear the gradients
        model.zero_grad()

        # Forward pass
        out_scores = model(feature_sequence)

        # Update parameters
        loss = loss_fn(out_scores, output_sequence)
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")

# See what the scores are after training
with torch.no_grad():
    total = 0
    correct = 0

    for feature_sequence, ground_truth in test_dataset:
        out_scores = model(feature_sequence)
        total += 1
        correct += int(torch.sum(torch.argmax(out_scores, dim=1) == ground_truth))

    print(f"Accuracy: {correct / total}")

# Save the model to models directory
models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "spacebar_detector_lstm.pth")
torch.save(model.state_dict(), model_path)
