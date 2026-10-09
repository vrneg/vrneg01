"""The unified motion/text vocabulary that makes motion "a foreign language".

MotionGPT's central move is to place motion codes and text tokens in one vocabulary so a
single sequence-to-sequence model can read and write both.  This module lays that
vocabulary out deterministically: motion codes keep the identifiers the VQ-VAE assigns,
and every special, task, and answer token is appended after them.

The text side is deliberately tiny.  The only natural language this task needs is a task
name and a two-word answer, so a pretrained subword tokenizer would add parameters
without adding information.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ALL_TASKS, ANSWER_WORDS

# Fixed specials, in the order they are appended after the motion codes.
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<eos>", "<bos>", "<som>", "<eom>")


@dataclass(frozen=True, slots=True)
class MotionLanguageVocabulary:
    """Identifier layout for motion codes, specials, task names, and answer words."""

    num_codes: int
    num_sentinels: int = 8
    task_names: tuple[str, ...] = ALL_TASKS
    answer_words: tuple[str, ...] = ANSWER_WORDS

    def __post_init__(self) -> None:
        if self.num_codes < 2:
            raise ValueError("num_codes must be at least 2")
        if self.num_sentinels < 1:
            raise ValueError("num_sentinels must be at least 1")
        if not self.task_names:
            raise ValueError("at least one task name is required")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be unique")
        if len(self.answer_words) < 2:
            raise ValueError("at least two answer words are required")
        if len(set(self.answer_words)) != len(self.answer_words):
            raise ValueError("answer_words must be unique")

    # -- layout ----------------------------------------------------------------

    @property
    def _special_base(self) -> int:
        return self.num_codes

    @property
    def _sentinel_base(self) -> int:
        return self._special_base + len(SPECIAL_TOKENS)

    @property
    def _task_base(self) -> int:
        return self._sentinel_base + self.num_sentinels

    @property
    def _answer_base(self) -> int:
        return self._task_base + len(self.task_names)

    @property
    def size(self) -> int:
        return self._answer_base + len(self.answer_words)

    # -- fixed specials --------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self._special_base + 0

    @property
    def eos_id(self) -> int:
        return self._special_base + 1

    @property
    def bos_id(self) -> int:
        return self._special_base + 2

    @property
    def start_of_motion_id(self) -> int:
        return self._special_base + 3

    @property
    def end_of_motion_id(self) -> int:
        return self._special_base + 4

    # -- parameterized tokens --------------------------------------------------

    def sentinel_id(self, index: int) -> int:
        """Return the identifier of the ``index``-th span-corruption sentinel."""

        if not 0 <= index < self.num_sentinels:
            raise ValueError(
                f"sentinel index {index} is outside [0, {self.num_sentinels})"
            )
        return self._sentinel_base + index

    def task_id(self, name: str) -> int:
        """Return the identifier of a task instruction token."""

        try:
            return self._task_base + self.task_names.index(name)
        except ValueError as error:
            raise ValueError(
                f"Unknown task {name!r}; expected one of {list(self.task_names)}"
            ) from error

    def answer_id(self, label: int) -> int:
        """Return the answer token for a class index."""

        if not 0 <= label < len(self.answer_words):
            raise ValueError(
                f"label {label} is outside [0, {len(self.answer_words)})"
            )
        return self._answer_base + label

    @property
    def answer_ids(self) -> tuple[int, ...]:
        return tuple(
            self._answer_base + index for index in range(len(self.answer_words))
        )

    def is_motion_token(self, token_id: int) -> bool:
        return 0 <= token_id < self.num_codes

    def token_name(self, token_id: int) -> str:
        """Return a printable name, for debugging built sequences."""

        if self.is_motion_token(token_id):
            return f"<motion_{token_id}>"
        if token_id < self._sentinel_base:
            return SPECIAL_TOKENS[token_id - self._special_base]
        if token_id < self._task_base:
            return f"<extra_id_{token_id - self._sentinel_base}>"
        if token_id < self._answer_base:
            return f"<task_{self.task_names[token_id - self._task_base]}>"
        if token_id < self.size:
            return self.answer_words[token_id - self._answer_base]
        raise ValueError(f"token id {token_id} is outside the vocabulary")

    def describe(self, token_ids: object) -> str:
        """Render a sequence of identifiers as readable token names."""

        if not isinstance(token_ids, (list, tuple)):
            token_ids = list(token_ids)  # type: ignore[arg-type]
        return " ".join(self.token_name(int(token_id)) for token_id in token_ids)


__all__ = ["SPECIAL_TOKENS", "MotionLanguageVocabulary"]
