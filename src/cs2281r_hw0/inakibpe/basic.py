from .base import Tokenizer, get_token_id_pair_counts, merge_token_ids


class BasicTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(
        self,
        text: str,
        max_vocab_size: int,
        verbose: bool = False,
    ):
        assert max_vocab_size >= 256
        max_num_merges = max_vocab_size - Tokenizer.intermediate_encoding_size

        new_token_ids = self._encode_to_intermediate(text)
        self.encoding_table: dict[tuple[int, int], int] = (
            {}
        )  # (original_id, original_id) -> new_token_id
        self.vocabulary: dict[int, bytes] = (
            self._generate_vocabulary()
        )  # Create empty vocabulary
        for i in range(max_num_merges):
            id_pair_counts = get_token_id_pair_counts(new_token_ids)
            max_occurring_pair = max(id_pair_counts, key=id_pair_counts.get)
            new_token_id = Tokenizer.intermediate_encoding_size + i
            self.encoding_table[max_occurring_pair] = new_token_id
            self.vocabulary[new_token_id] = (
                self.vocabulary[max_occurring_pair[0]]
                + self.vocabulary[max_occurring_pair[1]]
            )
            new_token_ids = merge_token_ids(
                new_token_ids, max_occurring_pair, new_token_id
            )
            if verbose:
                print(
                    f"Merging [{i+1}/{max_num_merges}]: ({max_occurring_pair[0]}, {max_occurring_pair[1]}) -> {new_token_id} ({id_pair_counts[max_occurring_pair]} occurrences)"
                )
        self.encoding_table

    def decode(self, token_ids: list[int]) -> str:
        intermediate_encoding_bytes = b"".join(
            [self.vocabulary[token_id] for token_id in token_ids]
        )  # Pythonic way of joining bytes - https://stackoverflow.com/a/17068310/7443346
        return self._decode_from_intermediate(intermediate_encoding_bytes)

    def encode(self, text: str) -> list[int]:
        encoded_tokens = self._encode_to_intermediate(text)
        while len(encoded_tokens) >= 2:
            pair_counts = get_token_id_pair_counts(encoded_tokens)
            lowest_id_pair = min(
                pair_counts, key=lambda p: self.encoding_table.get(p, float("inf"))
            )  # Pair with the lowest token_id (to ensure correct order of encoding)
            if lowest_id_pair not in self.encoding_table:
                break  # If the min pair is not in encoding table there are no more pairs to merge
            token_id = self.encoding_table[lowest_id_pair]
            encoded_tokens = merge_token_ids(encoded_tokens, lowest_id_pair, token_id)
        return encoded_tokens
