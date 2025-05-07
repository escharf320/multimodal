import signal
import sys
import torch
from num2words import num2words
from models import device, tokenizer, T5WithAdapter

model = T5WithAdapter(freeze_t5=True)
model = model.to(device)

# Handle Ctrl+C


def save_on_quit(sig, frame):
    print("\nSaving model before exiting...")
    torch.save(model.state_dict(), "t5_adapter_model_checkpoint.pth")
    print("Model saved. Exiting...")
    sys.exit(0)


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, save_on_quit)
print("Press Ctrl+C to save and exit")


train_dataset = []

for i in range(1, 1000):
    train_dataset.append((i, num2words(i)))

print(train_dataset[20:30])

test_dataset = [
    (11, "eleven"),
    (12, "twelve"),
    (13, "thirteen"),
    (14, "fourteen"),
    (15, "fifteen"),
    (16, "sixteen"),
    (17, "seventeen"),
]

epochs = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Set the model to train mode
model.train()

# Track losses for each epoch
epoch_losses = []

for epoch in range(epochs):
    epoch_loss = 0

    for number, output in train_dataset:
        # zero the gradients
        optimizer.zero_grad()  # Use optimizer.zero_grad() instead of model.zero_grad()

        # build the target encodings
        target_encodings = tokenizer(
            output,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # build the input feature
        input_feature = torch.zeros(1, 20).to(device)
        input_feature[0, 0] = number

        # forward pass - no need to calculate input_embeds and attention_mask separately
        outputs = model(
            time_series_features=input_feature,
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
    for number, output in [
        (-1, None),
        (500, None),
        (1500, None),
    ]:
        # build the input feature
        input_feature = torch.zeros(1, 20).to(device)
        input_feature[0, 0] = number

        # forward pass
        outputs = model.generate(input_feature)
        print(f"Input: {number}, Output: {outputs[0]}")

# Save the model

torch.save(model.state_dict(), "t5_adapter_model.pth")
