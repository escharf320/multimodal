import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================================
# > Hyperparameters
# ================================

# Adapater parameters
INPUT_DIM = 624
HIDDEN_DIM = 700
OUTPUT_DIM = 768

# LLM parameters
MODEL_NAME = "t5-base"

# Generation parameters
MAX_LENGTH = 64
NUM_BEAMS = 4
EARLY_STOPPING = True

# ================================
# > Setup
# ================================

# Set up device
DEVICE_IDENTIFIER = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device = torch.device(DEVICE_IDENTIFIER)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


# Define MLP adapter
class TimeSeriesAdapter(nn.Module):
    """
    MLP adapter for the LLM model.
    """

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.mlp(x)


# Define LLM-Adapter model
class LLMWithAdapter(nn.Module):
    """
    T5 model with MLP adapter.
    """

    def __init__(self, freeze_llm=False):
        super().__init__()

        self.adapter = TimeSeriesAdapter()
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Freeze the T5 model
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def embed_time_series_features(self, time_series_features):
        """
        Embed the time series features within instructions for the T5 model.
        """

        prefix = "USER: <h_start>"
        suffix = "<h_end> What did the user type? ASSISTANT:"
        ts_embeddings = self.adapter(time_series_features).view(
            1, time_series_features.size(0), -1
        )
        prefix_encodings = tokenizer(prefix, return_tensors="pt").to(device)
        prefix_embeds = self.llm.encoder.embed_tokens(prefix_encodings.input_ids)
        suffix_encodings = tokenizer(suffix, return_tensors="pt").to(device)
        suffix_embeds = self.llm.encoder.embed_tokens(suffix_encodings.input_ids)

        # Concatenate the time series embeddings with the token embeddings
        input_embeds = torch.cat([prefix_embeds, ts_embeddings, suffix_embeds], dim=1)

        # Update attention mask
        ts_mask = torch.ones(
            ts_embeddings.size()[:2],
            dtype=prefix_encodings.attention_mask.dtype,
            device=prefix_encodings.attention_mask.device,
        )
        attention_mask = torch.cat(
            [prefix_encodings.attention_mask, ts_mask, suffix_encodings.attention_mask],
            dim=1,
        )

        return input_embeds, attention_mask

    def forward(self, time_series_features, labels):
        """
        Forward pass for the T5-Adapter model.
        """

        # Prepare the input and attention mask
        input_embeds, attention_mask = self.embed_time_series_features(
            time_series_features
        )

        # Feed forward to T5
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, time_series_features):
        """
        Generate the output for the T5-Adapter model.
        """

        # Prepare the input and attention mask
        input_embeds, attention_mask = self.embed_time_series_features(
            time_series_features
        )

        # Generate the output
        generated_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
