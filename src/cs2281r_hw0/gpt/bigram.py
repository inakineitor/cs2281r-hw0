from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, cast
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, functional as F

from minbpe import RegexTokenizer

# ===========================
# ========== Types ==========
# ===========================

SplitsAvailable = Literal["training", "validation"]

# =====================================
# ========== Hyperparameters ==========
# =====================================

BATCH_SIZE = 64  # How many independent sequences will we process in parallel?
BLOCK_SIZE = 256  # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
NUM_EMBEDDING_DIMENSIONS = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT_PROBABILITY = 0.2
MODEL_WEIGHTS_PATH = "./model/shakespeare"

TRAIN_FRACTION = 0.9  # Remaining will be validation

DATASET_FILE = "./data/tinyshakespeare/input.txt"

# ===============================
# ========== Main Code ==========
# ===============================

torch.manual_seed(1337)  # TODO: Change this later


class Head(nn.Module):
    """
    One head of self-attenttion.
    """

    key_layer: nn.Linear
    query_layer: nn.Linear
    value_layer: nn.Linear
    triangular_mask: torch.Tensor

    def __init__(self, head_size: int):
        super().__init__()
        self.key_layer = nn.Linear(NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.query_layer = nn.Linear(NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.value_layer = nn.Linear(NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.register_buffer(
            "triangular_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        self.dropout_layer = nn.Dropout(DROPOUT_PROBABILITY)

    def forward(self, X: torch.Tensor):
        _, num_steps, num_channels = X.shape
        key = self.key_layer(X)  # (B, T, C)
        query = self.query_layer(X)  # (B, T, C)

        # Compute attention scores ("affinities").
        weights = (
            query @ key.transpose(-2, -1) * num_channels**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.triangular_mask[:num_steps, :num_steps] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout_layer(weights)

        # Perform the weighted aggregation of the values
        value = self.value_layer(X)  # (B, T, C)
        output = weights @ value  # (B, T, T) @ (B, T, C) ->(B, T, C)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    heads: nn.ModuleList
    projection_layer: nn.Linear

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection_layer = nn.Linear(
            NUM_EMBEDDING_DIMENSIONS, NUM_EMBEDDING_DIMENSIONS
        )
        self.dropout_layer = nn.Dropout(DROPOUT_PROBABILITY)

    def forward(self, X: torch.Tensor):
        output = torch.cat([head(X) for head in self.heads], dim=-1)
        output = self.dropout_layer(self.projection_layer(output))
        return output


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    net: nn.Sequential

    def __init__(self, num_dimensions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dimensions, 4 * num_dimensions),
            nn.ReLU(),
            nn.Linear(4 * NUM_EMBEDDING_DIMENSIONS, NUM_EMBEDDING_DIMENSIONS),
            nn.Dropout(DROPOUT_PROBABILITY),
        )

    def forward(self, X: torch.Tensor):
        return self.net(X)


class Block(nn.Module):
    """
    Tranformer block: Communication followed by computation.
    """

    self_attention_heads: MultiHeadAttention
    feed_forward_network: FeedForward
    layer_norm_1: nn.LayerNorm
    layer_norm_2: nn.LayerNorm

    def __init__(self, num_embedding_dimensions: int, num_heads: int):
        """
        :param num_embedding_dimensions: Number of embedding dimensions.
        :param num_heads: Number of heads.
        """
        super().__init__()
        head_size = num_embedding_dimensions // num_heads
        self.self_attention_heads = MultiHeadAttention(num_heads, head_size)
        self.feed_forward_network = FeedForward(num_embedding_dimensions)
        self.layer_norm_1 = nn.LayerNorm(num_embedding_dimensions)
        self.layer_norm_2 = nn.LayerNorm(num_embedding_dimensions)

    def forward(self, X: torch.Tensor):
        X = X + self.self_attention_heads(
            self.layer_norm_1(X)
        )  # TODO: Is it common in Python to override variables like this
        X = X + self.feed_forward_network(self.layer_norm_2(X))
        return X


# Super simple bigram model
class BigramLanguageModel(nn.Module):
    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    transformer_blocks: nn.Sequential
    language_model_head: nn.Linear

    def __init__(self, vocabulary_size: int):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(
            vocabulary_size, NUM_EMBEDDING_DIMENSIONS
        )
        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE, NUM_EMBEDDING_DIMENSIONS
        )
        self.transformer_blocks = nn.Sequential(
            *[
                Block(NUM_EMBEDDING_DIMENSIONS, num_heads=NUM_HEADS)
                for _ in range(NUM_LAYERS)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(NUM_EMBEDDING_DIMENSIONS)
        self.language_model_head = nn.Linear(NUM_EMBEDDING_DIMENSIONS, vocabulary_size)

    def forward(
        self, idx: torch.LongTensor, targets: Optional[torch.LongTensor] = None
    ):
        """
        idx: (BATCH_SIZE, T) tensor
        targets: (BATCH_SIZE, T) tensor
        """
        batch_size, num_time_steps = idx.shape

        token_embeddings: torch.Tensor = self.token_embedding_table(
            idx
        )  # (BATCH_SIZE, TIME, CHANNELS)
        position_embeddings: torch.Tensor = self.position_embedding_table(
            torch.arange(num_time_steps, device=DEVICE)
        )  # (TIME, CHANNEL)
        combined_embeddings = (
            token_embeddings + position_embeddings
        )  # (BATCH_SIZE, NUM_STEPS, CHANNELS)
        combined_embeddings = self.transformer_blocks(
            combined_embeddings
        )  # (BATCH_SIZE, NUM_STEPS, CHANNELS)
        combined_embeddings = self.final_layer_norm(
            combined_embeddings
        )  # (BATCH_SIZE, NUM_STEPS, CHANNELS)

        logits: torch.Tensor = self.language_model_head(
            combined_embeddings
        )  # (BATCH_SIZE, NUM_STEPS, vocabulary_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            batch_size, num_steps, num_channels = logits.shape
            reshaped_logits = logits.view(batch_size * num_steps, num_channels)
            reshaped_targets = targets.view(batch_size * num_steps)
            loss = F.cross_entropy(reshaped_logits, reshaped_targets)

        return logits, loss  # TODO: Review reshaping

    def generate(self, idx: torch.LongTensor, max_new_tokens: int) -> torch.LongTensor:
        """
        idx: (BATCH_SIZE, T) array of indices in the current context
        """
        for _ in range(max_new_tokens):
            cropped_idx = idx[
                :, -BLOCK_SIZE:
            ]  # Crop idx to the last `BLOCK_SIZE` tokens.
            logits, _ = self(cropped_idx)  # Get the predictions.
            last_time_logits = logits[
                :, -1, :
            ]  # Focus only on the last time step. Becomes (BATCH_SIZE, CHANNELS).
            probabilities = F.softmax(
                last_time_logits, dim=-1
            )  # Apply softmax to get probabilities. Becomes (BATCH_SIZE, CHANNELS).
            idx_next = torch.multinomial(
                probabilities, num_samples=1
            )  # Sample from the distribution. (B, 1).
            idx = cast(
                torch.LongTensor, torch.cat((idx, idx_next), dim=1)
            )  # Append sampled index to the running sequence. (BATCH_SIZE, TIME + 1)
        return idx


class CharacterLevelEncoder:
    def __init__(self, training_text: str):
        self.chars = sorted(list(set(training_text)))
        # Create a mapping from characters to integers.
        self.stoi = {char: num for num, char in enumerate(self.chars)}
        self.itos = {num: char for num, char in enumerate(self.chars)}

    def get_vocabulary_size(self) -> int:
        return len(self.chars)

    def encode(self, string: str) -> list[int]:
        return [
            self.stoi[char] for char in string
        ]  # Encoder: take a string, output a list of integers.

    def decode(self, token_list: list[int]) -> str:
        return "".join([self.itos[token] for token in token_list])


def train(dataset_text: str, encoder: CharacterLevelEncoder) -> nn.Module:
    # Generate train and test splits.
    all_data = torch.tensor(encoder.encode(dataset_text), dtype=torch.long)
    num_training_samples = int(TRAIN_FRACTION * len(all_data))
    training_data = all_data[:num_training_samples]
    validation_data = all_data[num_training_samples:]

    # Loading data.
    def get_data_batch(
        split: Literal["training", "validation"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate a small batch of data of inputs `X` and targets `Y`.
        # Format seen here: https://pasteboard.co/K9hceelsncwf.png
        split_data = training_data if split == "training" else validation_data
        X_indices = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
        X = torch.stack([split_data[idx : idx + BLOCK_SIZE] for idx in X_indices])
        X_in_device = X.to(DEVICE)
        Y = torch.stack(
            [split_data[idx + 1 : idx + BLOCK_SIZE + 1] for idx in X_indices]
        )
        Y_in_device = Y.to(DEVICE)

        return X_in_device, Y_in_device

    @torch.no_grad()
    def estimate_loss(model: BigramLanguageModel):
        output: dict[SplitsAvailable, float] = {}
        model.eval()  # TODO: Fix dependency injection
        splits_available: list[Literal["training", "validation"]] = [
            "training",
            "validation",
        ]
        for split in splits_available:
            losses = torch.zeros(EVAL_ITERS)
            for iteration in range(EVAL_ITERS):
                X, Y = get_data_batch(split)
                _, loss = model(X, Y)  # Equivalent to model.forward
                losses[iteration] = loss.item()
            output[split] = losses.mean().item()
        model.train()
        return output

    model = BigramLanguageModel(encoder.get_vocabulary_size())
    on_device_model = model.to(DEVICE)  # TODO: Move to device in same line

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )  # Create a PyTorch optimizer.

    for iter in range(MAX_ITERS):
        # Every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            training_loss = losses["training"]
            validation_loss = losses["validation"]
            print(f"===== Step {iter} =====")
            print(f"- Training loss: {training_loss:.4f}")
            print(f"- Validation loss: {validation_loss:.4f}")

        # Sample a batch of data
        XB, YB = get_data_batch("training")

        # Evaluate the loss
        _, loss = model(XB, YB)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return on_device_model


def load_dataset() -> str:
    dataset_text = ""
    with open(DATASET_FILE, "r", encoding="utf-8") as dataset_text_file:
        dataset_text = dataset_text_file.read()
    return dataset_text


def run_only():
    dataset_text = load_dataset()
    encoder = CharacterLevelEncoder(dataset_text)

    on_device_model = BigramLanguageModel(encoder.get_vocabulary_size())
    on_device_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
    on_device_model.to(DEVICE)

    # Generate text from the model
    context = cast(
        torch.LongTensor, torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    )
    print(
        encoder.decode(
            on_device_model.generate(context, max_new_tokens=500)[0].tolist()
        )
    )


def main():
    dataset_text = load_dataset()
    tokenizer = minib
    encoder = CharacterLevelEncoder(dataset_text)

    on_device_model = train(dataset_text, encoder)

    torch.save(on_device_model.state_dict(), MODEL_WEIGHTS_PATH)

    # Generate text from the model
    context = cast(
        torch.LongTensor, torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    )
    print(
        encoder.decode(
            on_device_model.generate(context, max_new_tokens=500)[0].tolist()
        )
    )


if __name__ == "__main__":
    main()
