from abc import ABC, abstractmethod
from collections import defaultdict
from regex import Pattern


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


class Tokenizer(ABC):
    # ========== Intermediate Encoding ==========
    intermediate_encoding = "utf-8"
    intermediate_encoding_size = 256  # Because each code unit in UTF-8 has 256 values

    def _encode_to_intermediate(self, text: str) -> list[int]:
        return list(text.encode(self.intermediate_encoding))

    def _decode_from_intermediate(self, intermediate_tokens: bytes | list[int]) -> str:
        decoded_text = bytes(intermediate_tokens).decode(
            self.intermediate_encoding, errors="replace"
        )  # Standard practice to not throw on invalid intermediate encoding byte sequence being output by model
        return decoded_text

    # ========== Tokenizer Encoding ==========
    encoding_table: dict[tuple[int, int], int]
    vocabulary: dict[int, bytes]
    regex_pattern: Pattern

    def __init__(
        self, encoding_table: dict[tuple[int, int], int] | None = None
    ) -> None:
        self.encoding_table = encoding_table if encoding_table is not None else {}
        self.vocabulary = self._generate_vocabulary()

    def _generate_vocabulary(self) -> dict[int, bytes]:
        vocabulary = {id: bytes([id]) for id in range(self.intermediate_encoding_size)}
        for original_token_ids, new_token_id in self.encoding_table.items():
            vocabulary[new_token_id] = (
                vocabulary[original_token_ids[0]] + vocabulary[original_token_ids[1]]
            )
        return vocabulary

    @abstractmethod
    def train(
        self,
        text: str,
        max_vocab_size: int,
        verbose: bool = False,
    ):
        """
        Train encoder from text.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
