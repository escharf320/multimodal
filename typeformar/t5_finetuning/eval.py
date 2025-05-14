import os
import torch
from tqdm import tqdm
from models import device
from train import load_model, CHECKPOINT_PATH
from dataset import partition_dataset

if not os.path.exists(CHECKPOINT_PATH):
    print("No model found, please train a model first")
    exit()

model = load_model()

# Load the test dataset

train_dataset, test_dataset = partition_dataset()

# Run evaluation

model.eval()

with torch.no_grad():
    total = 0
    correct = 0

    for word, feature_vectors in tqdm(test_dataset):
        total += 1

        # build the input feature
        time_series_features = torch.stack(feature_vectors).to(device)

        # forward pass
        outputs = model.generate(time_series_features)
        tqdm.write(f"Expected: {word}, Actual: {outputs[0]}")

        if word == outputs[0]:
            correct += 1

    print(f"Accuracy: {correct / total}")
