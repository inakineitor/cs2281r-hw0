from collections.abc import Callable
from typing import Literal, Optional, cast
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, functional as F

# ===========================
# ========== Types ==========
# ===========================

SplitsAvailable = Literal["training", "validation"]

# =====================================
# ========== Hyperparameters ==========
# =====================================

BATCH_SIZE = 30  # How many independent sequences will we process in parallel?
BLOCK_SIZE = 8  # What is the maximum context length for predictions?
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200

TRAIN_FRACTION = 0.9  # Remaining will be validation

DATASET_FILE = "./data/tinyshakespeare/input.txt"

# ===============================
# ========== Main Code ==========
# ===============================

torch.manual_seed(1337)  # TODO: Change this later


# Super simple bigram model
class BigramLanguageModel(nn.Module):
    token_embedding_table: nn.Embedding

    def __init__(self, vocabulary_size: int):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(
        self, idx: torch.LongTensor, targets: Optional[torch.LongTensor] = None
    ):
        """
        idx: (BATCH_SIZE, T) tensor
        targets: (BATCH_SIZE, T) tensor
        """
        logits: torch.Tensor = self.token_embedding_table(
            idx
        )  # (BATCH_SIZE, TIME, CHANNELS)

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
            logits, loss = self(idx)  # Get the predictions.
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


def main():
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dataset_text = ""
    with open(DATASET_FILE, "r", encoding="utf-8") as dataset_text_file:
        dataset_text = dataset_text_file.read()

    # Here are all the unique characters that occur in this text.
    chars = sorted(list(set(dataset_text)))
    vocabulary_size = len(chars)

    # Create a mapping from characters to integers.
    stoi = {char: num for num, char in enumerate(chars)}
    itos = {num: char for num, char in enumerate(chars)}

    def encode(string: str) -> list[int]:
        return [
            stoi[char] for char in string
        ]  # Encoder: take a string, output a list of integers.

    def decode(token_list: list[int]) -> str:
        return "".join([itos[token] for token in token_list])

    # Generate train and test splits.
    all_data = torch.tensor(encode(dataset_text), dtype=torch.long)
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

    model = BigramLanguageModel(vocabulary_size)
    m = model.to(DEVICE)  # TODO: Move to device in same line

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

    # Generate text from the model
    context = cast(
        torch.LongTensor, torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    )
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
