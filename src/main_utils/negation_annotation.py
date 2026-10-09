import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from dneg import Pipeline
from nltk.tokenize import PunktTokenizer
from surrealdb import RecordID, Surreal
from tqdm import tqdm


@lru_cache(maxsize=1)
def sdb_login():
    with open(Path(__file__).resolve().parent.parent.parent / "data/surreal/sdb_login.json", "r") as f:
        config = json.load(f)
    return config


class Transcript(TypedDict):
    transcript: list[str]
    word_ids: list[Any]


class SentencizedTranscript(TypedDict):
    transcript: list[list[str]]
    word_ids: list[list[Any]]


class Negation(TypedDict):
    cue_token_ids: list[tuple[int, int]]
    scope_token_ids: list[tuple[int, int]]


class NegationSentencizedTranscript(TypedDict):
    transcript: list[list[str]]
    word_ids: list[list[Any]]

    negations: list[Negation]


_UPLOAD_NEGATIONS_QUERY = """
BEGIN TRANSACTION;

REMOVE TABLE IF EXISTS negation;

DEFINE TABLE negation
    TYPE NORMAL
    SCHEMAFULL
    PERMISSIONS NONE;

DEFINE FIELD cue_tokens ON TABLE negation
    TYPE array<record<Word>>
    REFERENCE ON DELETE REJECT
    ASSERT $value.len() >= 0;

DEFINE FIELD scope_tokens ON TABLE negation
    TYPE array<record<Word>>
    REFERENCE ON DELETE REJECT
    ASSERT $value.len() >= 0;

INSERT INTO negation $negations RETURN NONE;

COMMIT TRANSACTION;
"""


_COMPOSITE_WORD_ID_PATTERN = re.compile(
    r'Word:\[\s*(-?\d+)\s*,\s*r"((?:\\.|[^"\\])*)"\s*,\s*'
    r'(-?\d+)\s*\]'
)
_PLAYER_ID_PATTERN = re.compile(
    r'Player:\[\s*(-?\d+)\s*,\s*s"((?:\\.|[^"\\])*)"\s*\]'
)


def _parse_word_record_id(word_id: str) -> RecordID:
    """Parse a stringified ``Word`` ID without changing its composite key.

    The SurrealDB Python SDK's ``RecordID.parse`` splits on every colon and
    cannot parse IDs such as ``Word:[..., r\"Player:[...]\", ...]``. Build the
    composite ID explicitly so that the SDK serializes its array and nested
    ``Player`` record with the correct CBOR types.
    """
    match = _COMPOSITE_WORD_ID_PATTERN.fullmatch(word_id)
    if match is None:
        return RecordID.parse(word_id)

    timestamp, escaped_player_id, token_index = match.groups()
    try:
        player_id = json.loads(f'"{escaped_player_id}"')
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid nested Player record ID in {word_id!r}") from error

    player_match = _PLAYER_ID_PATTERN.fullmatch(player_id)
    if player_match is None:
        raise ValueError(f"Invalid nested Player record ID in {word_id!r}")

    player_number, escaped_role = player_match.groups()
    try:
        role = json.loads(f'"{escaped_role}"')
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid nested Player record ID in {word_id!r}") from error

    return RecordID(
        "Word",
        [
            int(timestamp),
            RecordID("Player", [int(player_number), role]),
            int(token_index),
        ],
    )


def read_transcripts(data: Any) -> dict[Any, Transcript]:
    result: dict[Any, Transcript] = {}
    for audio in data:
        audio_id = audio["id"]
        transcript = []
        word_ids = []
        for word in audio["words"]:
            word_form = word["text"]
            word_id = word["id"]

            transcript.append(word_form)
            word_ids.append(word_id)
        assert len(transcript) == len(word_ids)
        result[audio_id] = Transcript(transcript=transcript,
                                      word_ids=word_ids)
    return result


def sentencize_transcripts(
    data: dict[Any, Transcript],
) -> dict[Any, SentencizedTranscript]:
    """Group each transcript and its word IDs using NLTK's German Punkt model.

    Punkt is run on a space-joined view of the transcript. Only the detected
    sentence spans are used: the returned values are slices of the original token
    and word-ID lists, so tokens are never changed, split, merged, or reordered.
    A new result mapping is returned; ``data`` itself is not modified.

    Requires NLTK's ``punkt_tab`` data package for the pretrained German model.
    """
    result: dict[Any, SentencizedTranscript] = {}
    sentence_tokenizer = load_sentence_tokenizer()

    for audio_id, audio in data.items():
        transcript = audio["transcript"]
        word_ids = audio["word_ids"]

        if len(transcript) != len(word_ids):
            raise ValueError(
                f"Transcript and word_ids for {audio_id!r} have different "
                f"lengths ({len(transcript)} != {len(word_ids)})"
            )

        text = " ".join(transcript)
        token_end_offsets: list[int] = []
        offset = 0
        for token in transcript:
            offset += len(token)
            token_end_offsets.append(offset)
            offset += 1  # Account for the joining space.

        sentence_end_indices: list[int] = []
        token_index = 0
        for _, sentence_end_offset in sentence_tokenizer.span_tokenize(text):
            while (
                token_index < len(token_end_offsets)
                and token_end_offsets[token_index] < sentence_end_offset
            ):
                token_index += 1

            sentence_end_index = min(token_index + 1, len(transcript))
            if not sentence_end_indices or sentence_end_index > sentence_end_indices[-1]:
                sentence_end_indices.append(sentence_end_index)

        if transcript and (
            not sentence_end_indices or sentence_end_indices[-1] < len(transcript)
        ):
            sentence_end_indices.append(len(transcript))

        transcript_sentences: list[list[str]] = []
        word_id_sentences: list[list[Any]] = []
        sentence_start_index = 0
        for sentence_end_index in sentence_end_indices:
            transcript_sentences.append(
                transcript[sentence_start_index:sentence_end_index]
            )
            word_id_sentences.append(
                word_ids[sentence_start_index:sentence_end_index]
            )
            sentence_start_index = sentence_end_index

        result[audio_id] = {
            "transcript": transcript_sentences,
            "word_ids": word_id_sentences,
        }

    return result


@lru_cache(maxsize=None)
def load_sentence_tokenizer() -> PunktTokenizer:
    """Load NLTK's pretrained German Punkt sentence tokenizer once."""
    return PunktTokenizer(lang="german")


@lru_cache(maxsize=None)
def load_dneg():
    pipe = Pipeline.from_language(lang="de",
                                  mode="gat",
                                  ds="sfu")
    return pipe


def annotate_negation(data: dict[Any, SentencizedTranscript]) -> dict[Any, NegationSentencizedTranscript]:
    pipe = load_dneg()
    result = {}
    for audio_key, audio in tqdm(data.items(), desc="Annotating negations"):
        res = pipe.run(audio["transcript"])
        negations = []
        for sent_idx in range(len(audio["transcript"])):
            for sent_neg in res[sent_idx]:
                ctok, stok = [], []
                for tok_idx, tok in enumerate(sent_neg):
                    if tok == "[CUE]":
                        ctok.append((sent_idx, tok_idx))
                    elif tok == "[SCO]":
                        stok.append((sent_idx, tok_idx))

                if not ctok:
                    raise ValueError(
                        "D-NEG returned a negation without a cue for "
                        f"audio {audio_key!r}, sentence {sent_idx}"
                    )

                neg = {
                    "cue_token_ids": ctok,
                    "scope_token_ids": stok,
                }
                negations.append(neg)
        result[audio_key] = {"transcript": audio["transcript"],
                             "word_ids": audio["word_ids"],
                             "negations": negations,}
        """print(*audio["transcript"])
        print(*audio["word_ids"])
        print(*negations)"""
    return result


def upload_negations(
    db: Any,
    data: dict[Any, NegationSentencizedTranscript],
) -> Any:
    """Upload annotations as records containing links to their ``Word`` tokens.

    ``db`` must be an authenticated, synchronous SurrealDB Python client with
    the desired namespace and database already selected. The schema and all
    negation records are written in one transaction. Each ``(sentence, token)``
    coordinate emitted by :func:`annotate_negation` is resolved through the
    corresponding nested ``word_ids`` list before the query is submitted.
    """

    records = []
    for audio_id, audio in data.items():
        word_ids = audio["word_ids"]

        def resolve_word_ids(
            positions: list[tuple[int, int]],
            negation_index: int,
            field_name: str,
        ) -> list[Any]:
            resolved = []
            for sentence_index, token_index in positions:
                if sentence_index < 0 or token_index < 0:
                    raise ValueError(
                        f"Invalid {field_name} coordinate "
                        f"({sentence_index}, {token_index}) in negation "
                        f"{negation_index} of audio {audio_id!r}"
                    )
                try:
                    word_id = word_ids[sentence_index][token_index]
                except IndexError as error:
                    raise ValueError(
                        f"Invalid {field_name} coordinate "
                        f"({sentence_index}, {token_index}) in negation "
                        f"{negation_index} of audio {audio_id!r}"
                    ) from error

                if isinstance(word_id, RecordID):
                    record_id = word_id
                elif isinstance(word_id, str):
                    try:
                        record_id = _parse_word_record_id(word_id)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid SurrealDB record ID {word_id!r} in "
                            f"audio {audio_id!r}"
                        ) from error
                else:
                    raise TypeError(
                        f"Expected a string or RecordID for audio {audio_id!r}, "
                        f"got {type(word_id).__name__}"
                    )

                if record_id.table_name != "Word":
                    raise ValueError(
                        f"Expected a Word record ID for audio {audio_id!r}, "
                        f"got {record_id}"
                    )
                resolved.append(record_id)

            return resolved

        for negation_index, negation in enumerate(audio["negations"]):
            records.append(
                {
                    "cue_tokens": resolve_word_ids(
                        negation["cue_token_ids"], negation_index, "cue"
                    ),
                    "scope_tokens": resolve_word_ids(
                        negation["scope_token_ids"], negation_index, "scope"
                    ),
                }
            )

    return db.query(_UPLOAD_NEGATIONS_QUERY, {"negations": records})


if __name__ == "__main__":
    with open(Path(__file__).resolve().parent.parent.parent / "data/surreal/test_data.json", "r") as f:
        dd = read_transcripts(json.load(f)[0])
        dd = sentencize_transcripts(dd)
        dd = annotate_negation(dd)
        with Surreal(sdb_login()["url"]) as db:
            db.signin({"username": sdb_login()["user"], "password": sdb_login()["pwd"]})
            db.use(sdb_login()["ns"], sdb_login()["db"])
            upload_negations(db, dd)
