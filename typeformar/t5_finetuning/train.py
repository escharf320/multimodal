import os
import signal
import sys
import torch
from num2words import num2words
from models import device, tokenizer, LLMWithAdapter
from dataset import partition_dataset

# Checkpoints
CHECKPOINT_PATH = "adapter_model_checkpoint.pth"


def save_model(model):
    torch.save(model.state_dict(), CHECKPOINT_PATH)


def load_model():
    model = LLMWithAdapter(freeze_llm=False)
    model = model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    return model


def load_model_if_it_exists():
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading model from checkpoint")
        return load_model()
    else:
        print("No checkpoint found, creating new model")
        model = LLMWithAdapter(freeze_llm=False)
        model = model.to(device)
        return model


# Load the model from checkpoint if it exists
model = load_model_if_it_exists()

# Handle Ctrl+C


def save_on_quit(sig, frame):
    print("\nSaving model before exiting...")
    save_model(model)
    print("Model saved. Exiting...")
    sys.exit(0)


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, save_on_quit)
print("Press Ctrl+C to save and exit")

train_dataset, test_dataset = partition_dataset()

epochs = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Set the model to train mode
model.train()

# Track losses for each epoch
epoch_losses = []

for epoch in range(epochs):
    epoch_loss = 0

    for word, feature_vectors in train_dataset:
        # zero the gradients
        optimizer.zero_grad()  # Use optimizer.zero_grad() instead of model.zero_grad()

        # build the target encodings
        target_encodings = tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        # build the input feature
        time_series_features = torch.stack(feature_vectors).to(device)

        # forward pass - no need to calculate input_embeds and attention_mask separately
        outputs = model(
            time_series_features=time_series_features,
            labels=target_encodings.input_ids,
        )

        # compute the loss
        loss = outputs.loss
        epoch_loss += loss.item()

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_dataset)
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

# Evaluate the model

model.eval()

with torch.no_grad():
    for word, feature_vectors in test_dataset:
        # build the input feature
        time_series_features = torch.stack(feature_vectors).to(device)

        # forward pass
        outputs = model.generate(time_series_features)
        print(f"Expected: {word}, Actual: {outputs[0]}")

# Save the model

torch.save(model.state_dict(), "adapter_model.pth")
