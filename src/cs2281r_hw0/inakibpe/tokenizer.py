from pathlib import Path
from collections import defaultdict
import typer


def get_token_id_pair_counts(token_ids: list[int]):
    id_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for pair in zip(token_ids, token_ids[1:]):
        id_pair_counts[pair] += 1
    return id_pair_counts


def merge_token_ids(token_ids: list[int], pair: tuple[int, int], new_token_id: int):
    new_token_ids = []
    idx = 0
    while idx < len(token_ids):
        # If we are the start of the desired pair, replace and skip to the following character
        if (
            idx < len(token_ids) - 1
            and token_ids[idx] == pair[0]
            and token_ids[idx + 1] == pair[1]
        ):
            new_token_ids.append(new_token_id)
            idx += 2
        else:
            new_token_ids.append(token_ids[idx])
            idx += 1
    return new_token_ids


def get_token_from_original_encoding(
    original_encoding_ids: list[int], max_vocab_size: int
):
    original_encoding_size = 256  # UTF-8 Uses 256 possible values in a unit
    num_merges = max_vocab_size - original_encoding_size

    new_token_ids = list(original_encoding_ids)
    merges: dict[tuple[int, int], int] = (
        {}
    )  # (original_id, original_id) -> new_token_id
    for i in range(num_merges):
        id_pair_counts = get_token_id_pair_counts(new_token_ids)
        max_occurring_pair = max(id_pair_counts, key=id_pair_counts.get)
        new_token_id = original_encoding_size + i
        merges[max_occurring_pair] = new_token_id
        new_token_ids = merge_token_ids(new_token_ids, max_occurring_pair, new_token_id)
        print(
            f"Merging ({max_occurring_pair[0]}, {max_occurring_pair[1]}) into {new_token_id}"
        )
    return new_token_ids, merges


class DigramTokenizer:
    encoding_table: dict[int, tuple[int, int]]

    def __init__(self, encoding_table: dict[int, tuple[int, int]]) -> None:
        self.encoding_table = encoding_table

    @staticmethod
    def create_from_encoding_table(encoding_table: dict[int, tuple[int, int]]):
        return DigramTokenizer(encoding_table)

    @staticmethod
    def create_from_encoded_text(
        original_encoding_ids: list[int],
        original_encoding_size: int,
        max_vocab_size: int,
    ):
        num_merges = max_vocab_size - original_encoding_size

        new_token_ids = list(original_encoding_ids)
        merges: dict[tuple[int, int], int] = (
            {}
        )  # (original_id, original_id) -> new_token_id
        for i in range(num_merges):
            id_pair_counts = get_token_id_pair_counts(new_token_ids)
            max_occurring_pair = max(id_pair_counts, key=id_pair_counts.get)
            new_token_id = original_encoding_size + i
            merges[max_occurring_pair] = new_token_id
            new_token_ids = merge_token_ids(
                new_token_ids, max_occurring_pair, new_token_id
            )
            print(
                f"Merging ({max_occurring_pair[0]}, {max_occurring_pair[1]}) into {new_token_id}"
            )
        return new_token_ids, merges

    @staticmethod
    def create_from_text(text: str, max_vocab_size: int):
        raw_bytes = text.encode("utf-8")
        original_encoding_size = 256  # For UTF-8
        original_encoding_ids = list(map(int, raw_bytes))
        return DigramTokenizer.create_from_encoded_text(
            original_encoding_ids, original_encoding_size, max_vocab_size
        )

    def encode_text(self, text: str) -> list[int]:
        pass

    def decode(self, token_ids: list[int]) -> list[int]:
        decoded_output = []
        for token_id in token_ids:
            assert token_id in self.encoding_table
            decoded_output.append()

    def decode_to_text(self, token_ids: list[int]) -> str:
        pass


def main(dataset_path: Path):
    dataset_text = ""
    with open(dataset_path, "r", encoding="utf-8") as input_file:
        dataset_str = input_file.read()

    # TODO: Remove. For debugging purposes only.
    dataset_text = dataset_str[:1000]

    dataset_raw_bytes = dataset_text.encode("utf-8")
    dataset_tokens = list(map(int, dataset_raw_bytes))

    print("---")
    print(dataset_text)
    print(f"length: {len(dataset_text)}")
    print("---")
    print(dataset_tokens)
    print(f"length: {len(dataset_tokens)}")

    merges = make_tokenizer(dataset_text, 270)


if __name__ == "__main__":
    typer.run(main)
