import json
import random
import time
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Literal, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import websockets
import websockets.sync.client as ws_sync
from huggingface_hub import HfApi, login

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Json,
    Value,
    concatenate_datasets,
)
from sklearn.model_selection import StratifiedGroupKFold
from surrealdb.connections.blocking_ws import BlockingWsSurrealConnection
from surrealdb.data.cbor import decode
from surrealdb.errors import ConnectionUnavailableError, UnexpectedResponseError
from surrealdb.request_message.message import RequestMessage
from tqdm import tqdm
from websockets.exceptions import WebSocketException

try:
    from .windowing import context_window_bounds_ms, encode_signed_milliseconds
except ImportError:  # Direct execution from ``src/main_utils``.
    from windowing import context_window_bounds_ms, encode_signed_milliseconds


@dataclass(frozen=True, slots=True)
class SurrealClientConfig:
    """Timeout and retry policy for read-only dataset extraction queries.

    ``max_attempts`` includes the initial attempt. Setting a timeout to ``None``
    disables that particular timeout, matching the websockets API.
    """

    open_timeout_seconds: float | None = 30.0
    query_timeout_seconds: float | None = 300.0
    close_timeout_seconds: float | None = 10.0
    ping_interval_seconds: float | None = 20.0
    ping_timeout_seconds: float | None = 20.0
    max_attempts: int = 8
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.5

    def validate(self) -> None:
        for name in (
            "open_timeout_seconds",
            "query_timeout_seconds",
            "close_timeout_seconds",
            "ping_interval_seconds",
            "ping_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer of at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be at least initial_backoff_seconds"
            )
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be non-negative")


# Edit this policy once to change all calls in this module, or pass an explicit
# SurrealClientConfig to create_dataset/select_words/select_contexts.
SURREAL_CLIENT_CONFIG = SurrealClientConfig()

# Script-wide control selection policy. ``coarse`` preserves the historical
# random/distant selection exactly. Change this to ``strict`` or
# ``very_strict`` or ``neg`` before running the script, or pass control_mode
# explicitly to create_dataset/select_words. ``neg`` pairs each cue with a word
# from that cue's annotated negation scope.
ControlSelectionMode = Literal["coarse", "strict", "very_strict", "neg"]
AnchorSelectionMode = Literal[
    "all",
    "coarse",
    "strict",
    "very_strict",
    "neg",
]
CONTROL_SELECTION_MODE: ControlSelectionMode = "strict"


def _resolve_selection_modes(
    control_mode: ControlSelectionMode | None,
    anchor_mode: AnchorSelectionMode | None,
) -> tuple[ControlSelectionMode, ControlSelectionMode]:
    """Resolve omitted modes so either side alone selects a matched dataset."""

    if control_mode is not None and control_mode not in {
        "coarse",
        "strict",
        "very_strict",
        "neg",
    }:
        raise ValueError(
            "control_mode must be 'coarse', 'strict', 'very_strict', or 'neg'"
        )
    if anchor_mode is not None and anchor_mode not in {
        "all",
        "coarse",
        "strict",
        "very_strict",
        "neg",
    }:
        raise ValueError(
            "anchor_mode must be 'coarse', 'strict', 'very_strict', or 'neg'"
        )

    normalized_anchor = "coarse" if anchor_mode == "all" else anchor_mode
    if control_mode is None and normalized_anchor is None:
        return CONTROL_SELECTION_MODE, CONTROL_SELECTION_MODE
    if control_mode is None:
        assert normalized_anchor is not None
        return normalized_anchor, normalized_anchor
    if normalized_anchor is None:
        return control_mode, control_mode
    return normalized_anchor, control_mode


class _TimeoutBlockingWsSurrealConnection(BlockingWsSurrealConnection):
    """SurrealDB's blocking WebSocket connection with configurable timeouts."""

    def __init__(self, url: str, config: SurrealClientConfig) -> None:
        super().__init__(url)
        self.config = config

    def _open_socket(self):
        return ws_sync.connect(
            self.raw_url,
            max_size=None,
            subprotocols=[websockets.Subprotocol("cbor")],
            open_timeout=self.config.open_timeout_seconds,
            close_timeout=self.config.close_timeout_seconds,
            ping_interval=self.config.ping_interval_seconds,
            ping_timeout=self.config.ping_timeout_seconds,
        )

    def _send(
        self,
        message: RequestMessage,
        process: str,
        bypass: bool = False,
    ) -> dict[str, Any]:
        # This mirrors surrealdb's BlockingWsSurrealConnection._send while adding
        # the receive timeout that its public constructor currently doesn't expose.
        with self._lock:
            if self.socket is None:
                self.socket = self._open_socket()
            self.socket.send(message.WS_CBOR_DESCRIPTOR)
            data = self.socket.recv(timeout=self.config.query_timeout_seconds)
            response = decode(data if isinstance(data, bytes) else data.encode())

            response_id = response.get("id")
            if response_id is not None and response_id != message.id:
                raise UnexpectedResponseError(
                    f"Response ID mismatch: expected {message.id}, got "
                    f"{response_id}."
                )
            if not bypass:
                self.check_response_for_error(response, process)
            return response

    def __enter__(self) -> "_TimeoutBlockingWsSurrealConnection":
        if self.socket is None:
            self.socket = self._open_socket()
        return self


_TRANSIENT_SURREAL_ERRORS = (
    TimeoutError,
    OSError,
    ConnectionUnavailableError,
    UnexpectedResponseError,
    WebSocketException,
)


class RetryingSurrealClient:
    """Reconnect and retry idempotent SurrealDB reads after transient outages."""

    def __init__(
        self,
        login_config: dict[str, Any],
        config: SurrealClientConfig | None = None,
        *,
        connection_factory: Callable[
            [str, SurrealClientConfig], BlockingWsSurrealConnection
        ] = _TimeoutBlockingWsSurrealConnection,
    ) -> None:
        self.login_config = dict(login_config)
        self.config = SURREAL_CLIENT_CONFIG if config is None else config
        self.config.validate()
        required = {"url", "user", "pwd", "ns", "db"}
        missing = sorted(required - self.login_config.keys())
        if missing:
            raise ValueError(
                "SurrealDB login configuration is missing: " + ", ".join(missing)
            )
        scheme = urlsplit(str(self.login_config["url"])).scheme.lower()
        if scheme not in {"ws", "wss"}:
            raise ValueError(
                "RetryingSurrealClient requires a ws:// or wss:// SurrealDB URL"
            )
        self._connection_factory = connection_factory
        self._connection: BlockingWsSurrealConnection | None = None

    def _discard_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            pass

    def close(self) -> None:
        self._discard_connection()

    def _connect_once(self) -> BlockingWsSurrealConnection:
        connection = self._connection_factory(
            str(self.login_config["url"]),
            self.config,
        )
        try:
            connection.__enter__()
            connection.signin(
                {
                    "username": self.login_config["user"],
                    "password": self.login_config["pwd"],
                }
            )
            connection.use(
                str(self.login_config["ns"]),
                str(self.login_config["db"]),
            )
        except BaseException:
            try:
                connection.close()
            except Exception:
                pass
            raise
        self._connection = connection
        return connection

    def _delay_seconds(self, failed_attempt: int) -> float:
        base_delay = min(
            self.config.max_backoff_seconds,
            self.config.initial_backoff_seconds
            * self.config.backoff_multiplier ** (failed_attempt - 1),
        )
        return base_delay + random.uniform(0.0, self.config.jitter_seconds)

    def _run_with_retry(
        self,
        operation: Callable[[BlockingWsSurrealConnection], Any],
        description: str,
    ) -> Any:
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                connection = self._connection or self._connect_once()
                return operation(connection)
            except _TRANSIENT_SURREAL_ERRORS as error:
                self._discard_connection()
                if attempt == self.config.max_attempts:
                    error.add_note(
                        f"SurrealDB {description} failed after "
                        f"{self.config.max_attempts} attempts"
                    )
                    raise
                delay = self._delay_seconds(attempt)
                print(
                    f"SurrealDB {description} attempt {attempt}/"
                    f"{self.config.max_attempts} failed with "
                    f"{type(error).__name__}: {error}. Retrying in "
                    f"{delay:.1f}s ...",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable SurrealDB retry state")

    def connect(self) -> None:
        self._run_with_retry(lambda connection: None, "connection")

    def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Run and retry an idempotent query after reconnecting and signing in."""

        return self._run_with_retry(
            lambda connection: connection.query(query, variables),
            "read query",
        )

    def __enter__(self) -> "RetryingSurrealClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@lru_cache(maxsize=1)
def sdb_login():
    login_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data/surreal/sdb_login.json"
    )
    with open(login_path, "r") as f:
        config = json.load(f)
    return config


def surreal_client(
    config: SurrealClientConfig | None = None,
) -> RetryingSurrealClient:
    """Create an authenticated, retrying client from the local login file."""

    return RetryingSurrealClient(sdb_login(), config)


_NEGATION_REFERENCE_PROJECTION = "<~negation.* AS neg"
_CUE_REFERENCE_PREDICATE = (
    "array::len(<~(negation FIELD cue_tokens)) > 0"
)
_SCOPE_REFERENCE_PREDICATE = (
    "array::len(<~(negation FIELD scope_tokens)) > 0"
)
_NEGATION_WORD_QUERIES = {
    "cue": (
        f"SELECT *, {_NEGATION_REFERENCE_PROJECTION} FROM Word "
        f"WHERE {_CUE_REFERENCE_PREDICATE}"
    ),
    "scope": (
        f"SELECT *, {_NEGATION_REFERENCE_PROJECTION} FROM Word "
        f"WHERE {_SCOPE_REFERENCE_PREDICATE}"
    ),
    "cueandscope": (
        f"SELECT *, {_NEGATION_REFERENCE_PROJECTION} FROM Word "
        f"WHERE ({_CUE_REFERENCE_PREDICATE}) "
        f"OR ({_SCOPE_REFERENCE_PREDICATE})"
    ),
}
_NON_NEGATION_WORD_QUERY = (
    f"SELECT *, {_NEGATION_REFERENCE_PROJECTION} FROM Word "
    "WHERE array::len(<~(negation FIELD cue_tokens)) = 0 "
    "AND array::len(<~(negation FIELD scope_tokens)) = 0"
)
_TURN_WORD_QUERY = "SELECT id, experiment, player, text, timeMs FROM Word"


def _word_sort_key(word: dict[str, Any]) -> tuple[str, int, str]:
    """Return a stable order independent of SurrealDB result ordering."""

    return (
        str(word.get("experiment", "")),
        int(word.get("timeMs", 0)),
        str(word.get("id", "")),
    )


def _assert_unique_word_ids(words: list[dict[str, Any]], name: str) -> None:
    word_ids = [str(word.get("id")) for word in words]
    if len(word_ids) != len(set(word_ids)):
        raise ValueError(f"{name} contains duplicate Word record IDs")


def _assert_disjoint_word_ids(
    first_words: list[dict[str, Any]],
    second_words: list[dict[str, Any]],
    names: str,
) -> None:
    first_ids = {str(word.get("id")) for word in first_words}
    second_ids = {str(word.get("id")) for word in second_words}
    overlap = first_ids & second_ids
    if overlap:
        raise ValueError(
            f"{names} contain {len(overlap)} overlapping Word record IDs"
        )


def _referenced_word_id(reference: Any) -> str | None:
    """Normalize a SurrealDB Word reference or an expanded record."""

    if isinstance(reference, Mapping):
        reference = reference.get("id")
    if reference is None:
        return None
    return str(reference)


def _scope_word_ids(positive: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the scope Word IDs attached to a cue through negation records."""

    negation_records = positive.get("neg")
    if not isinstance(negation_records, (list, tuple)):
        return ()

    scope_ids: set[str] = set()
    for negation_record in negation_records:
        if not isinstance(negation_record, Mapping):
            continue
        scope_tokens = negation_record.get("scope_tokens")
        if not isinstance(scope_tokens, (list, tuple)):
            continue
        for scope_token in scope_tokens:
            scope_id = _referenced_word_id(scope_token)
            if scope_id is not None:
                scope_ids.add(scope_id)
    return tuple(sorted(scope_ids))


_NEG_CONTROL_MIN_CUE_DISTANCE_MS = 1_100
_STRICT_CONTROL_FIELDS = ("experiment", "player", "chunk")
_STRICT_CONTROL_MIN_CUE_DISTANCE_MS = 1_000
_STRICT_CONTROL_EXPANSION_STEP_MS = 10_000


def _same_speaker_turns(
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split chronological experiment words when the speaking player changes."""

    turns: list[list[dict[str, Any]]] = []
    current_turn: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    for word in sorted(words, key=_word_sort_key):
        experiment = word.get("experiment")
        player = word.get("player")
        if experiment is None or player is None:
            raise ValueError(
                f"Word {word.get('id')!r} is missing a very-strict turn field"
            )
        _strict_control_timestamp_ms(word)
        key = (
            json.dumps(experiment, sort_keys=True, default=str),
            json.dumps(player, sort_keys=True, default=str),
        )
        if current_turn and key != current_key:
            turns.append(current_turn)
            current_turn = []
        current_turn.append(word)
        current_key = key
    if current_turn:
        turns.append(current_turn)
    return turns


def _turn_position_bins(
    turns: list[list[dict[str, Any]]],
) -> dict[str, Literal["early", "middle", "late"]]:
    """Assign each word to a third of its same-speaker turn by word order."""

    bins: dict[str, Literal["early", "middle", "late"]] = {}
    for turn in turns:
        final_index = len(turn) - 1
        for index, word in enumerate(turn):
            # A one-word turn has no temporal extent, so its only word is the
            # natural midpoint. Longer turns place their first and last words
            # exactly at 0 and 1, respectively.
            if final_index == 0:
                position = 0.5
            else:
                position = index / final_index
            if position < 1 / 3:
                position_bin = "early"
            elif position < 2 / 3:
                position_bin = "middle"
            else:
                position_bin = "late"
            bins[str(word.get("id"))] = position_bin
    return bins


@lru_cache(maxsize=1)
def _german_pos_tagger():
    """Load spaCy lazily so existing control modes pay no NLP startup cost."""

    import spacy

    try:
        return spacy.load(
            "de_core_news_sm",
            disable=["parser", "ner", "lemmatizer"],
        )
    except OSError as error:
        raise RuntimeError(
            "very_strict control mode requires the spaCy de_core_news_sm model"
        ) from error


def _coarse_pos_by_word_id(
    turns: list[list[dict[str, Any]]],
) -> dict[str, str]:
    """Contextually annotate database words with spaCy's coarse Universal POS."""

    turn_texts: list[str] = []
    character_spans: list[list[tuple[int, int]]] = []
    for turn in turns:
        pieces: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for word in turn:
            text = word.get("text")
            if not isinstance(text, str) or not text.strip():
                text = " "
            if pieces:
                cursor += 1
            start = cursor
            pieces.append(text)
            cursor += len(text)
            spans.append((start, cursor))
        turn_texts.append(" ".join(pieces))
        character_spans.append(spans)

    tags: dict[str, str] = {}
    nlp = _german_pos_tagger()
    documents = nlp.pipe(turn_texts, batch_size=128)
    for turn, spans, document in zip(turns, character_spans, documents):
        for word, (start, end) in zip(turn, spans):
            overlapping_tokens = [
                token
                for token in document
                if token.idx < end and token.idx + len(token.text) > start
            ]
            lexical_tokens = [
                token
                for token in overlapping_tokens
                if not token.is_space and not token.is_punct
            ]
            usable_tokens = lexical_tokens or [
                token for token in overlapping_tokens if not token.is_space
            ]
            if usable_tokens and usable_tokens[0].pos_:
                tags[str(word.get("id"))] = usable_tokens[0].pos_
    return tags


def _strict_control_group_key(
    word: Mapping[str, Any],
) -> tuple[str, str, str]:
    values = []
    for field in _STRICT_CONTROL_FIELDS:
        if field not in word or word[field] is None:
            raise ValueError(
                f"Word {word.get('id')!r} is missing strict-control field "
                f"{field!r}"
            )
        values.append(json.dumps(word[field], sort_keys=True, default=str))
    return values[0], values[1], values[2]


def _interview_speaker_control_group_key(
    word: Mapping[str, Any],
) -> tuple[str, str]:
    values = []
    for field in _STRICT_CONTROL_FIELDS[:2]:
        if field not in word or word[field] is None:
            raise ValueError(
                f"Word {word.get('id')!r} is missing very-strict-control "
                f"field {field!r}"
            )
        values.append(json.dumps(word[field], sort_keys=True, default=str))
    return values[0], values[1]


def _strict_control_timestamp_ms(word: Mapping[str, Any]) -> int:
    value = word.get("timeMs")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Word {word.get('id')!r} has non-integer strict-control timeMs"
        )
    return value


def _strict_fallback_radius_ms(nearest_distance_ms: int) -> int:
    """Return the first 10-second search radius containing the distance."""

    steps = (
        nearest_distance_ms + _STRICT_CONTROL_EXPANSION_STEP_MS - 1
    ) // _STRICT_CONTROL_EXPANSION_STEP_MS
    return max(
        _STRICT_CONTROL_EXPANSION_STEP_MS,
        steps * _STRICT_CONTROL_EXPANSION_STEP_MS,
    )


def _neg_control_timestamp_ms(word: Mapping[str, Any]) -> int:
    value = word.get("timeMs")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Word {word.get('id')!r} has non-integer neg-control timeMs"
        )
    return value


def _select_neg_controls(
    negation_words: list[dict[str, Any]],
    scope_words: list[dict[str, Any]],
    balance: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair every retainable cue with one word from its own negation scope.

    Scope words are sampled reproducibly and without replacement. Every scope
    onset must be at least 1100 ms from its cue onset, keeping a +/-1000 ms
    scope-centered window clear of the cue onset. A cue is skipped when none of
    its attached negation records contains an eligible scope word. Cue Word IDs
    are never reused as controls, including when a Word is annotated as both a
    cue and part of a scope.
    """

    positive_fraction = 0.5 if balance is None else balance
    if positive_fraction != 0.5:
        raise ValueError(
            "neg control mode requires balance=None or balance=0.5"
        )

    positive_ids = {str(word.get("id")) for word in negation_words}
    scope_words_by_id = {
        str(word.get("id")): word
        for word in scope_words
        if str(word.get("id")) not in positive_ids
    }
    positive_timestamps = {
        str(word.get("id")): _neg_control_timestamp_ms(word)
        for word in negation_words
    }
    scope_timestamps = {
        scope_id: _neg_control_timestamp_ms(scope_word)
        for scope_id, scope_word in scope_words_by_id.items()
    }
    candidate_ids = {
        str(positive.get("id")): [
            scope_id
            for scope_id in _scope_word_ids(positive)
            if scope_id in scope_words_by_id
            and abs(
                scope_timestamps[scope_id]
                - positive_timestamps[str(positive.get("id"))]
            ) >= _NEG_CONTROL_MIN_CUE_DISTANCE_MS
        ]
        for positive in negation_words
    }

    rng = random.Random(42)
    matching_order = list(negation_words)
    rng.shuffle(matching_order)
    matching_order.sort(
        key=lambda positive: len(candidate_ids[str(positive.get("id"))])
    )
    available_scope_ids = set(scope_words_by_id)
    selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for positive in matching_order:
        eligible_scope_ids = [
            scope_id
            for scope_id in candidate_ids[str(positive.get("id"))]
            if scope_id in available_scope_ids
        ]
        if not eligible_scope_ids:
            continue
        selected_scope_id = rng.choice(eligible_scope_ids)
        available_scope_ids.remove(selected_scope_id)
        selected_pairs.append(
            (positive, scope_words_by_id[selected_scope_id])
        )

    selected_positives = [positive for positive, _ in selected_pairs]
    selected_controls = [control for _, control in selected_pairs]
    if not selected_positives:
        raise ValueError(
            "No negation cues have an available word in their annotated scope "
            "at least 1100 ms from cue onset"
        )

    selected_positives.sort(key=_word_sort_key)
    selected_controls.sort(key=_word_sort_key)
    _assert_unique_word_ids(selected_positives, "neg positives")
    _assert_unique_word_ids(selected_controls, "neg scope controls")
    _assert_disjoint_word_ids(
        selected_positives,
        selected_controls,
        "neg positives and scope controls",
    )
    if len(selected_positives) != len(selected_controls):
        raise RuntimeError("Neg control selection produced unbalanced classes")
    print(
        "Neg control selection retained "
        f"{len(selected_positives)}/{len(negation_words)} cues with one "
        "unique word from the cue's annotated negation scope at least 1100 ms "
        "from cue onset."
    )
    return selected_positives, selected_controls


def _select_coarse_controls(
    distance_reference_words: list[dict[str, Any]],
    non_negation_words: list[dict[str, Any]],
    min_distance_ms: int,
    select_n: int,
    positive_count: int,
) -> list[dict[str, Any]]:
    """Sample unique controls using the historical global-distance rule."""

    negation_timestamps = sorted(
        word["timeMs"] for word in distance_reference_words
    )
    distant_non_negation_words = []
    for word in tqdm(non_negation_words, desc="Selecting non negation words"):
        timestamp = word["timeMs"]
        insertion_index = bisect_left(negation_timestamps, timestamp)
        distances = []
        if insertion_index > 0:
            distances.append(
                abs(timestamp - negation_timestamps[insertion_index - 1])
            )
        if insertion_index < len(negation_timestamps):
            distances.append(
                abs(timestamp - negation_timestamps[insertion_index])
            )
        if not distances or min(distances) >= min_distance_ms:
            distant_non_negation_words.append(word)

    if select_n > len(distant_non_negation_words):
        raise ValueError(
            f"Requested {select_n} non-negation controls for "
            f"{positive_count} positive examples, but only "
            f"{len(distant_non_negation_words)} satisfy min_distance_ms="
            f"{min_distance_ms}"
        )

    rng = random.Random(42)
    selected_controls = sorted(
        rng.sample(distant_non_negation_words, k=select_n),
        key=_word_sort_key,
    )
    _assert_unique_word_ids(selected_controls, "selected controls")
    return selected_controls


def _select_strict_controls(
    negation_words: list[dict[str, Any]],
    non_negation_words: list[dict[str, Any]],
    balance: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair cues with distant controls, preferring their exact audio chunk.

    Every selected control is at least one second from its cue onset. Exact
    experiment/player/chunk candidates are tried first. If none are available,
    the chunk constraint is relaxed while retaining the experiment and player,
    and the temporal search radius grows as +/-10 s, +/-20 s, and so on until a
    candidate becomes available. Controls are sampled without replacement.
    """

    positive_fraction = 0.5 if balance is None else balance
    if positive_fraction != 0.5:
        raise ValueError(
            "strict control mode currently requires balance=None or balance=0.5"
        )

    rng = random.Random(42)
    positive_metadata = {
        str(word.get("id")): (
            _strict_control_group_key(word),
            _strict_control_timestamp_ms(word),
        )
        for word in negation_words
    }
    control_metadata = {
        str(word.get("id")): (
            _strict_control_group_key(word),
            _strict_control_timestamp_ms(word),
        )
        for word in non_negation_words
    }
    controls_by_interview_speaker: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}
    for control in non_negation_words:
        control_group, _ = control_metadata[str(control.get("id"))]
        controls_by_interview_speaker.setdefault(control_group[:2], []).append(
            control
        )

    available_control_ids = {
        str(control.get("id")) for control in non_negation_words
    }

    def eligible_controls(
        positive: dict[str, Any],
        *,
        same_chunk_only: bool,
    ) -> list[tuple[dict[str, Any], int]]:
        positive_group, positive_time = positive_metadata[str(positive.get("id"))]
        candidates = []
        for control in controls_by_interview_speaker.get(positive_group[:2], []):
            control_id = str(control.get("id"))
            if control_id not in available_control_ids:
                continue
            control_group, control_time = control_metadata[control_id]
            if same_chunk_only and control_group != positive_group:
                continue
            distance_ms = abs(control_time - positive_time)
            if distance_ms >= _STRICT_CONTROL_MIN_CUE_DISTANCE_MS:
                candidates.append((control, distance_ms))
        return candidates

    selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    matched_positive_ids: set[str] = set()
    same_chunk_matches = 0
    fallback_matches = 0
    maximum_fallback_radius_ms = 0

    # Complete exact-chunk matching before any expanded-window control can
    # consume a control belonging to another cue's preferred audio chunk.
    same_chunk_order = list(negation_words)
    rng.shuffle(same_chunk_order)
    same_chunk_order.sort(
        key=lambda positive: len(
            eligible_controls(positive, same_chunk_only=True)
        )
    )
    for positive in same_chunk_order:
        positive_id = str(positive.get("id"))
        candidates = eligible_controls(positive, same_chunk_only=True)
        if not candidates:
            continue
        selected_control, _ = rng.choice(candidates)
        available_control_ids.remove(str(selected_control.get("id")))
        matched_positive_ids.add(positive_id)
        selected_pairs.append((positive, selected_control))
        same_chunk_matches += 1

    # Randomize equal-opportunity fallback cues reproducibly, but try cues with
    # fewer remaining controls first so flexible cues do not consume scarce ones.
    fallback_order = [
        positive
        for positive in negation_words
        if str(positive.get("id")) not in matched_positive_ids
    ]
    rng.shuffle(fallback_order)
    fallback_order.sort(
        key=lambda positive: len(
            eligible_controls(positive, same_chunk_only=False)
        )
    )
    for positive in fallback_order:
        candidates = eligible_controls(positive, same_chunk_only=False)
        if not candidates:
            continue
        nearest_distance_ms = min(distance_ms for _, distance_ms in candidates)
        fallback_radius_ms = _strict_fallback_radius_ms(nearest_distance_ms)
        fallback_controls = [
            control
            for control, distance_ms in candidates
            if distance_ms <= fallback_radius_ms
        ]
        selected_control = rng.choice(fallback_controls)
        available_control_ids.remove(str(selected_control.get("id")))
        selected_pairs.append((positive, selected_control))
        fallback_matches += 1
        maximum_fallback_radius_ms = max(
            maximum_fallback_radius_ms,
            fallback_radius_ms,
        )

    selected_positives = [positive for positive, _ in selected_pairs]
    selected_controls = [control for _, control in selected_pairs]

    if not selected_positives:
        raise ValueError(
            "No strict cue/control pairs share an experiment and player while "
            "remaining at least 1000 ms apart"
        )
    selected_positives.sort(key=_word_sort_key)
    selected_controls.sort(key=_word_sort_key)
    _assert_unique_word_ids(selected_positives, "strict positives")
    _assert_unique_word_ids(selected_controls, "strict controls")
    _assert_disjoint_word_ids(
        selected_positives,
        selected_controls,
        "strict positives and controls",
    )
    if len(selected_positives) != len(selected_controls):
        raise RuntimeError("Strict control selection produced unbalanced classes")
    print(
        "Strict control selection retained "
        f"{len(selected_positives)}/{len(negation_words)} positives with one "
        "same-experiment/player control each at least 1 s from cue onset "
        f"({same_chunk_matches} same-chunk, {fallback_matches} expanded-window; "
        f"maximum fallback radius {maximum_fallback_radius_ms / 1000:g} s)."
    )
    return selected_positives, selected_controls


def _select_very_strict_controls(
    negation_words: list[dict[str, Any]],
    non_negation_words: list[dict[str, Any]],
    turn_words: list[dict[str, Any]],
    balance: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair cues without replacement on context, coarse POS, and turn third.

    Candidates must come from the same experiment and player, but may come from
    a different audio chunk. They must also have the same spaCy Universal POS
    tag and early/middle/late same-speaker-turn bin as their cue. There is no
    minimum temporal distance in this mode.
    """

    positive_fraction = 0.5 if balance is None else balance
    if positive_fraction != 0.5:
        raise ValueError(
            "very_strict control mode requires balance=None or balance=0.5"
        )

    turns = _same_speaker_turns(turn_words)
    position_bins = _turn_position_bins(turns)
    coarse_pos_tags = _coarse_pos_by_word_id(turns)

    def matching_key(
        word: dict[str, Any],
    ) -> tuple[tuple[str, str], str, str] | None:
        word_id = str(word.get("id"))
        coarse_pos = coarse_pos_tags.get(word_id)
        position_bin = position_bins.get(word_id)
        if coarse_pos is None or position_bin is None:
            return None
        return (
            _interview_speaker_control_group_key(word),
            coarse_pos,
            position_bin,
        )

    positive_metadata = {
        str(word.get("id")): matching_key(word)
        for word in negation_words
    }
    control_metadata = {
        str(word.get("id")): matching_key(word)
        for word in non_negation_words
    }
    controls_by_matching_key: dict[
        tuple[tuple[str, str], str, str],
        list[dict[str, Any]],
    ] = {}
    for control in non_negation_words:
        control_key = control_metadata[str(control.get("id"))]
        if control_key is not None:
            controls_by_matching_key.setdefault(control_key, []).append(control)

    available_control_ids = {
        str(control.get("id")) for control in non_negation_words
    }

    def eligible_controls(
        positive: dict[str, Any],
    ) -> list[dict[str, Any]]:
        positive_key = positive_metadata[str(positive.get("id"))]
        if positive_key is None:
            return []
        candidates = []
        for control in controls_by_matching_key.get(positive_key, []):
            control_id = str(control.get("id"))
            if control_id not in available_control_ids:
                continue
            candidates.append(control)
        return candidates

    rng = random.Random(42)
    matching_order = list(negation_words)
    rng.shuffle(matching_order)
    matching_order.sort(key=lambda positive: len(eligible_controls(positive)))

    selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for positive in matching_order:
        candidates = eligible_controls(positive)
        if not candidates:
            continue
        selected_control = rng.choice(candidates)
        available_control_ids.remove(str(selected_control.get("id")))
        selected_pairs.append((positive, selected_control))

    selected_positives = [positive for positive, _ in selected_pairs]
    selected_controls = [control for _, control in selected_pairs]
    if not selected_positives:
        raise ValueError(
            "No very-strict cue/control pairs share an experiment/player and "
            "coarse POS and turn-position bin"
        )

    selected_positives.sort(key=_word_sort_key)
    selected_controls.sort(key=_word_sort_key)
    _assert_unique_word_ids(selected_positives, "very-strict positives")
    _assert_unique_word_ids(selected_controls, "very-strict controls")
    _assert_disjoint_word_ids(
        selected_positives,
        selected_controls,
        "very-strict positives and controls",
    )
    if len(selected_positives) != len(selected_controls):
        raise RuntimeError(
            "Very-strict control selection produced unbalanced classes"
        )
    print(
        "Very-strict control selection retained "
        f"{len(selected_positives)}/{len(negation_words)} positives with one "
        "unique control each, matched on experiment/player, coarse POS, "
        "and early/middle/late turn position."
    )
    return selected_positives, selected_controls


def _require_exact_anchor_set(
    expected_positives: list[dict[str, Any]],
    matched_positives: list[dict[str, Any]],
    control_mode: ControlSelectionMode,
) -> None:
    expected_ids = {str(word.get("id")) for word in expected_positives}
    matched_ids = {str(word.get("id")) for word in matched_positives}
    if expected_ids != matched_ids:
        missing_count = len(expected_ids - matched_ids)
        raise ValueError(
            f"{control_mode} controls are unavailable for {missing_count}/"
            f"{len(expected_ids)} fixed positive anchors; refusing to shrink "
            "the comparison anchor set"
        )


def select_words(
    negation_target: Literal["cue", "scope", "cueandscope"],
    min_distance_ms: int = 20000,
    balance: Optional[float] = None,
    surreal_client_config: SurrealClientConfig | None = None,
    control_mode: ControlSelectionMode | None = None,
    anchor_mode: AnchorSelectionMode | None = None,
):
    resolved_anchor_mode, resolved_control_mode = _resolve_selection_modes(
        control_mode,
        anchor_mode,
    )
    if min_distance_ms < 0:
        raise ValueError("min_distance_ms must be non-negative")
    if balance is not None and not 0.0 < balance <= 1.0:
        raise ValueError("balance must be in (0, 1]")
    if (
        resolved_control_mode == "strict" or resolved_anchor_mode == "strict"
    ) and balance not in {None, 0.5}:
        raise ValueError(
            "strict control mode currently requires balance=None or balance=0.5"
        )
    if (
        resolved_control_mode == "very_strict"
        or resolved_anchor_mode == "very_strict"
    ) and balance not in {None, 0.5}:
        raise ValueError(
            "very_strict control mode requires balance=None or balance=0.5"
        )
    if (
        resolved_control_mode == "neg" or resolved_anchor_mode == "neg"
    ) and balance not in {None, 0.5}:
        raise ValueError(
            "neg control mode requires balance=None or balance=0.5"
        )
    if (
        resolved_control_mode == "neg" or resolved_anchor_mode == "neg"
    ) and negation_target != "cue":
        raise ValueError(
            "neg control mode requires negation_target='cue' because its "
            "positive anchors must be negation cues"
        )

    try:
        negation_query = _NEGATION_WORD_QUERIES[negation_target]
    except KeyError as error:
        raise NotImplementedError(f">{negation_target}<") from error

    with surreal_client(surreal_client_config) as db:
        # A wildcard reverse traversal contains entries for both reference fields
        # and may interleave real records with NONE placeholders. Filtering on
        # ``neg[0]`` therefore silently omitted about half of the annotations.
        # Field-specific reverse-reference predicates select every linked Word.
        negation_words = sorted(
            db.query(negation_query),
            key=_word_sort_key,
        )
        needs_non_negation_words = any(
            mode != "neg"
            for mode in (resolved_anchor_mode, resolved_control_mode)
        )
        non_negation_words = (
            sorted(
                db.query(_NON_NEGATION_WORD_QUERY),
                key=_word_sort_key,
            )
            if needs_non_negation_words
            else []
        )
        needs_scope_words = (
            resolved_control_mode == "neg" or resolved_anchor_mode == "neg"
        )
        scope_words = (
            sorted(
                db.query(_NEGATION_WORD_QUERIES["scope"]),
                key=_word_sort_key,
            )
            if needs_scope_words
            else []
        )
        _assert_unique_word_ids(negation_words, "negation_words")
        _assert_unique_word_ids(non_negation_words, "non_negation_words")
        _assert_unique_word_ids(scope_words, "scope_words")
        _assert_disjoint_word_ids(
            negation_words,
            non_negation_words,
            "negation_words and non_negation_words",
        )
        distance_reference_words = negation_words

        turn_words = None
        if (
            resolved_control_mode == "very_strict"
            or resolved_anchor_mode == "very_strict"
        ):
            # Fetch the complete chronological word stream only for this opt-in
            # mode. Cue-only and non-negation query results omit scope-only words,
            # which would otherwise create false turn boundaries/positions.
            turn_words = sorted(
                db.query(_TURN_WORD_QUERY),
                key=_word_sort_key,
            )
            _assert_unique_word_ids(turn_words, "turn_words")

        anchor_controls: list[dict[str, Any]] | None = None
        if resolved_anchor_mode == "strict":
            negation_words, anchor_controls = _select_strict_controls(
                negation_words,
                non_negation_words,
                balance,
            )
        elif resolved_anchor_mode == "very_strict":
            assert turn_words is not None
            negation_words, anchor_controls = _select_very_strict_controls(
                negation_words,
                non_negation_words,
                turn_words,
                balance,
            )
        elif resolved_anchor_mode == "neg":
            negation_words, anchor_controls = _select_neg_controls(
                negation_words,
                scope_words,
                balance,
            )

        if (
            resolved_control_mode == resolved_anchor_mode
            and anchor_controls is not None
        ):
            return negation_words, anchor_controls

        if resolved_control_mode == "strict":
            matched_positives, selected_controls = _select_strict_controls(
                negation_words,
                non_negation_words,
                balance,
            )
            _require_exact_anchor_set(
                negation_words,
                matched_positives,
                resolved_control_mode,
            )
        elif resolved_control_mode == "very_strict":
            assert turn_words is not None
            matched_positives, selected_controls = _select_very_strict_controls(
                negation_words,
                non_negation_words,
                turn_words,
                balance,
            )
            _require_exact_anchor_set(
                negation_words,
                matched_positives,
                resolved_control_mode,
            )
        elif resolved_control_mode == "neg":
            matched_positives, selected_controls = _select_neg_controls(
                negation_words,
                scope_words,
                balance,
            )
            _require_exact_anchor_set(
                negation_words,
                matched_positives,
                resolved_control_mode,
            )
        else:
            positive_fraction = 0.5 if balance is None else balance
            select_n = int(
                (len(negation_words) - positive_fraction * len(negation_words))
                / positive_fraction
            )
            selected_controls = _select_coarse_controls(
                distance_reference_words,
                non_negation_words,
                min_distance_ms,
                select_n,
                len(negation_words),
            )
            _assert_disjoint_word_ids(
                negation_words,
                selected_controls,
                "coarse positives and controls",
            )

    if resolved_control_mode != resolved_anchor_mode:
        print(
            f"{resolved_control_mode} control selection retained the exact "
            f"{len(negation_words)} {resolved_anchor_mode} positive anchors."
        )

    return negation_words, selected_controls


def select_contexts(
    words: Any,
    window_ms_left: int = 1000,
    window_ms_right: int = 1000,
    event_source: Literal["speaker", "listener", "both"] = "speaker",
    surreal_client_config: SurrealClientConfig | None = None,
):
    window_start_ms, window_end_ms = context_window_bounds_ms(
        window_ms_left,
        window_ms_right,
    )
    if event_source == "speaker":
        context_query = """SELECT
    *,
    id[0] AS timestamp,
    record::tb(id) AS event_type
FROM
    Eye:[$from]..[$to],
    RightHand:[$from]..[$to],
    LeftHand:[$from]..[$to],
    Facial:[$from]..[$to],
    RightFinger:[$from]..[$to],
    LeftFinger:[$from]..[$to],
    Body:[$from]..[$to],
    Head:[$from]..[$to]
WHERE player = $playerid
ORDER BY timestamp ASC"""
    elif event_source == "listener":
        context_query = """SELECT
    *,
    id[0] AS timestamp,
    record::tb(id) AS event_type
FROM
    Eye:[$from]..[$to],
    RightHand:[$from]..[$to],
    LeftHand:[$from]..[$to],
    Facial:[$from]..[$to],
    RightFinger:[$from]..[$to],
    LeftFinger:[$from]..[$to],
    Body:[$from]..[$to],
    Head:[$from]..[$to]
WHERE player != $playerid
ORDER BY timestamp ASC"""
    elif event_source == "both":
        context_query = """SELECT
    *,
    id[0] AS timestamp,
    record::tb(id) AS event_type
FROM
    Eye:[$from]..[$to],
    RightHand:[$from]..[$to],
    LeftHand:[$from]..[$to],
    Facial:[$from]..[$to],
    RightFinger:[$from]..[$to],
    LeftFinger:[$from]..[$to],
    Body:[$from]..[$to],
    Head:[$from]..[$to]
ORDER BY timestamp ASC"""
    else:
        raise NotImplementedError(">" + event_source + "<")
    contexts = []
    with surreal_client(surreal_client_config) as db:
        for word in tqdm(words, desc="Selecting contexts"):
            context_events = db.query(context_query,
                     {"from": word["timeMs"] + window_start_ms,
                      # "to": word["timeMs"] + word["duration"] * 1000 + window_ms,
                      "to": word["timeMs"] + window_end_ms,
                      "playerid": word["player"]})
            # print(len(context_events))
            # print(context_events[0])
            contexts.append(context_events)
    return contexts


def dataset_from_list_chunked(
        rows: list[dict[str, Any]],
        features: Features,
        chunk_size: int = 25,
) -> Dataset:
    """Build a dataset without fingerprinting all variable-length data at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunks = []
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        print(f"Creating Arrow chunk {start}:{end} of {len(rows)}")
        chunks.append(Dataset.from_list(rows[start:end], features=features))

    if not chunks:
        return Dataset.from_list(rows, features=features)
    if len(chunks) == 1:
        return chunks[0]
    return concatenate_datasets(chunks)


def _dataset_prefix(
    negation_target: str,
    event_source: str,
    window_ms_left: int,
    window_ms_right: int,
    n_splits: int,
    balance: float | None,
    control_mode: ControlSelectionMode | None = None,
    anchor_mode: AnchorSelectionMode | None = None,
) -> str:
    context_window_bounds_ms(window_ms_left, window_ms_right)
    balance_label = "05" if balance is None else f"{balance}".replace(".", "")
    left_label = encode_signed_milliseconds(window_ms_left)
    right_label = encode_signed_milliseconds(window_ms_right)
    prefix = (
        f"tgt-{negation_target}_bal-{balance_label}_src-{event_source}_"
        f"wL-{left_label}_wR-{right_label}_spl-{n_splits}"
    )
    resolved_anchor_mode, resolved_control_mode = _resolve_selection_modes(
        control_mode,
        anchor_mode,
    )
    if resolved_anchor_mode != resolved_control_mode:
        anchor_label = resolved_anchor_mode.replace("_", "-")
        control_label = resolved_control_mode.replace("_", "-")
        prefix += f"_ctrl-{control_label}_anchors-{anchor_label}"
    elif resolved_control_mode == "very_strict":
        prefix += "_ctrl-very-strict"
    elif resolved_control_mode == "neg":
        prefix += "_ctrl-neg"
    elif resolved_control_mode == "strict" and anchor_mode == "strict":
        # The new anchor-only spelling gets a safe name. The historical
        # control_mode="strict" spelling retains its old unsuffixed name.
        prefix += "_ctrl-strict"
    return prefix


def create_dataset(
    negation_target: Literal["cue", "scope", "cueandscope"] = "cue",
    event_source: Literal["speaker", "listener", "both"] = "speaker",
    min_distance_ms: int = 20000,
    window_ms_left: int = 1000,
    window_ms_right: int = 1000,
    n_splits: int = 10,
    upload: bool = True,
    chunked: bool = True,
    chunk_size: int = 25,
    balance: Optional[float] = None,
    surreal_client_config: SurrealClientConfig | None = None,
    control_mode: ControlSelectionMode | None = None,
    anchor_mode: AnchorSelectionMode | None = None,
):
    context_window_bounds_ms(window_ms_left, window_ms_right)
    neg_words, non_neg_words = select_words(
        negation_target=negation_target,
        min_distance_ms=min_distance_ms,
        balance=balance,
        surreal_client_config=surreal_client_config,
        control_mode=control_mode,
        anchor_mode=anchor_mode,
    )
    non_neg_contexts = select_contexts(
        non_neg_words,
        window_ms_left=window_ms_left,
        window_ms_right=window_ms_right,
        event_source=event_source,
        surreal_client_config=surreal_client_config,
    )
    neg_contexts = select_contexts(
        neg_words,
        window_ms_left=window_ms_left,
        window_ms_right=window_ms_right,
        event_source=event_source,
        surreal_client_config=surreal_client_config,
    )
    assert len(non_neg_contexts) == len(non_neg_words)
    assert len(neg_contexts) == len(neg_words)
    ds = []
    for sample in zip(non_neg_contexts, non_neg_words):
        ds.append({"context": sample[0], "word": sample[1], "label": "none"})
    for sample in zip(neg_contexts, neg_words):
        ds.append({"context": sample[0], "word": sample[1], "label": "neg"})
    features = Features({
        "context": Json(),
        "word": Json(),
        "label": Value("string")
    })
    if chunked:
        ds = dataset_from_list_chunked(
            ds,
            features=features,
            chunk_size=chunk_size,
        )
    else:
        ds = Dataset.from_list(ds, features=features)
    create_dataset_folds(ds=ds,
                         output_dir=Path(__file__).resolve().parent.parent.parent / "data/trainsets",
                         n_splits=n_splits,
                         seed=42,
                         prefix=_dataset_prefix(
                             negation_target=negation_target,
                             event_source=event_source,
                             window_ms_left=window_ms_left,
                             window_ms_right=window_ms_right,
                             n_splits=n_splits,
                             balance=balance,
                             control_mode=control_mode,
                             anchor_mode=anchor_mode,
                         ),
                         upload=upload)
    return ds


def create_dataset_all_words(
        upload: bool = True,
        chunked: bool = True,
        chunk_size: int = 25,
        windows: Tuple[Tuple[int, int]] = ((500, 500), (1000, 1000)),
        event_source: Literal["speaker", "listener", "both"] = "speaker",
        surreal_client_config: SurrealClientConfig | None = None,
                    ):

    with surreal_client(surreal_client_config) as db:
        words = db.query("SELECT * from Word")

    for (wl, wr) in windows:
        context_window_bounds_ms(wl, wr)
        ctx = select_contexts(
            words,
            window_ms_left=wl,
            window_ms_right=wr,
            event_source=event_source,
            surreal_client_config=surreal_client_config,
        )
        ds = []
        for sample in zip(ctx, words):
            ds.append({"context": sample[0], "word": sample[1], "label": "none"})
        features = Features({
            "context": Json(),
            "word": Json(),
            "label": Value("string")
        })
        if chunked:
            ds = dataset_from_list_chunked(
                ds,
                features=features,
                chunk_size=chunk_size,
            )
        else:
            ds = Dataset.from_list(ds, features=features)
        output_dir = Path(__file__).resolve().parent.parent.parent / "data/trainsets_tokenizer"
        left_label = encode_signed_milliseconds(wl)
        right_label = encode_signed_milliseconds(wr)
        prefix = (
            f"target-allWords_source-{event_source}_"
            f"windowL-{left_label}_windowR-{right_label}"
        )
        try:
            ds.save_to_disk(str(output_dir / prefix))
        except Exception as e:
            print(f"Dataset saving failed with {e}")
        try:
            if upload:
                with open(Path(__file__).resolve().parent.parent.parent / "data/hf/login_token.json", "r") as f:
                    login_data = json.load(f)
                    login(token=login_data["token"])
                ds.push_to_hub(
                    f"{login_data['org_name']}/{prefix}",
                    private=True,
                )
        except Exception as e:
            print(f"Download failed: {e}")


def create_dataset_folds(
        ds,
        output_dir: str | Path,
        prefix: str,
        n_splits: int = 10,
        seed: int = 42,
        upload: bool = True,
):
    labels = np.asarray(ds["label"])

    # Use experiment if it represents one recording/session.
    groups = np.asarray([
        str(word["experiment"])
        for word in ds["word"]
    ])

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    partitions = [
        test_indices
        for _, test_indices in splitter.split(
            X=np.zeros(len(ds)),
            y=labels,
            groups=groups,
        )
    ]

    output_dir = Path(output_dir)
    folds = []

    for fold_number in range(n_splits):
        test_partition = fold_number
        validation_partition = (fold_number + 1) % n_splits
        train_partitions = [
            i for i in range(n_splits)
            if i not in {test_partition, validation_partition}
        ]

        train_indices = np.concatenate([
            partitions[i] for i in train_partitions
        ])
        validation_indices = partitions[validation_partition]
        test_indices = partitions[test_partition]

        fold = DatasetDict({
            "train": ds.select(train_indices.tolist()),
            "validation": ds.select(validation_indices.tolist()),
            "test": ds.select(test_indices.tolist()),
        })

        # Verify that no experiment crosses split boundaries.
        split_groups = {
            name: {
                str(word["experiment"])
                for word in split["word"]
            }
            for name, split in fold.items()
        }
        assert split_groups["train"].isdisjoint(split_groups["validation"])
        assert split_groups["train"].isdisjoint(split_groups["test"])
        assert split_groups["validation"].isdisjoint(split_groups["test"])

        fold.save_to_disk(str(output_dir / f"{prefix}_fold-{fold_number}"))
        if upload:
            with open(Path(__file__).resolve().parent.parent.parent / "data/hf/login_token.json", "r") as f:
                login_data = json.load(f)
                login(token=login_data["token"])
            fold.push_to_hub(
                f"{login_data['org_name']}/{prefix}_fold-{fold_number}",
                private=True,
            )
        folds.append(fold)

    return folds


def _default_dataset_configs() -> list[dict[str, Any]]:
    event_sources = ["speaker", "listener", "both"]
    window_sizes = [100, 250, 500, 1000, 1500, 2000, 2500, 5000]
    dataset_configs: list[dict[str, Any]] = []

    # Pre-anchor windows
    for event_source in event_sources:
        for window_size in window_sizes:
            dataset_configs.append({
                "negation_target": "cue",
                "event_source": event_source,
                "window_ms_left": window_size,
                "window_ms_right": 0,
                "chunk_size": 250
            })

    # Post-anchor windows
    for event_source in event_sources:
        for window_size in window_sizes:
            dataset_configs.append({
                "negation_target": "cue",
                "event_source": event_source,
                "window_ms_left": 0,
                "window_ms_right": window_size,
                "chunk_size": 250
            })

    return dataset_configs


def _sliding_window_dataset_configs() -> list[dict[str, Any]]:
    """Return contiguous 500 ms cue windows spanning -2500 through +2500 ms.

    ``window_ms_left`` is subtracted from the anchor and ``window_ms_right``
    is added, so the physical interval ``[-2500, -2000]`` is represented by
    ``(2500, -2000)`` and named ``wL-2500_wR-m2000``.  Ten intervals are
    generated for each event source.
    """

    dataset_configs: list[dict[str, Any]] = []
    for event_source in ("speaker", "listener", "both"):
        for window_start_ms in range(-2500, 2500, 500):
            window_end_ms = window_start_ms + 500
            dataset_configs.append({
                "negation_target": "cue",
                "event_source": event_source,
                "window_ms_left": -window_start_ms,
                "window_ms_right": window_end_ms,
                "chunk_size": 250,
            })
    return dataset_configs


def _expected_hub_dataset_ids(
    config: Mapping[str, Any],
    organization: str,
) -> set[str]:
    n_splits = int(config.get("n_splits", 10))
    prefix = _dataset_prefix(
        negation_target=config.get("negation_target", "cue"),
        event_source=config.get("event_source", "speaker"),
        window_ms_left=int(config.get("window_ms_left", 1000)),
        window_ms_right=int(config.get("window_ms_right", 1000)),
        n_splits=n_splits,
        balance=config.get("balance"),
        control_mode=config.get("control_mode"),
        anchor_mode=config.get("anchor_mode"),
    )
    return {
        f"{organization}/{prefix}_fold-{fold_number}"
        for fold_number in range(n_splits)
    }


def _partition_dataset_configs(
    dataset_configs: list[dict[str, Any]],
    organization: str,
    hub_dataset_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split configs by whether all of their fold repositories exist."""

    existing = []
    missing = []
    for config in dataset_configs:
        anchor_mode, control_mode = _resolve_selection_modes(
            config.get("control_mode"),
            config.get("anchor_mode"),
        )
        if (
            control_mode == "strict"
            and anchor_mode == "strict"
            and config.get("anchor_mode") is None
        ):
            # Historical strict datasets share repository names with coarse
            # datasets, so name-only Hub inventory cannot distinguish them.
            missing.append(config)
            continue
        expected_ids = _expected_hub_dataset_ids(config, organization)
        destination = existing if expected_ids <= hub_dataset_ids else missing
        destination.append(config)
    return existing, missing


def _hub_dataset_inventory() -> tuple[str, set[str]]:
    """Return the configured organization and its authenticated dataset IDs."""

    credentials_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data/hf/login_token.json"
    )
    with credentials_path.open("r", encoding="utf-8") as file:
        login_data = json.load(file)

    organization = login_data["org_name"]
    api = HfApi(token=login_data["token"])
    dataset_ids = {
        dataset.id for dataset in api.list_datasets(author=organization)
    }
    return organization, dataset_ids


def _create_configured_datasets(
    dataset_configs: list[dict[str, Any]],
) -> None:
    """Create missing configured datasets after one Hub inventory preflight."""

    organization, hub_dataset_ids = _hub_dataset_inventory()
    existing_configs, missing_configs = _partition_dataset_configs(
        dataset_configs,
        organization,
        hub_dataset_ids,
    )

    print("\nHugging Face Hub configuration preflight")
    print(f"Already found on Hub: {len(existing_configs)}")
    print(f"Still missing: {len(missing_configs)}")

    finished = []
    failed = []

    for config in missing_configs:
        name = (
            f"target={config['negation_target']}, "
            f"source={config['event_source']}, "
            f"window=({config['window_ms_left']}, {config['window_ms_right']})"
        )

        try:
            print(f"\nCreating: {name}")

            result = create_dataset(**config)
            print(result)

            finished.append(name)

        except Exception as exc:
            print(f"FAILED: {name}")
            print(f"Error: {type(exc).__name__}: {exc}")

            failed.append({
                "dataset": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    print("\n" + "=" * 80)
    print("DATASET CREATION SUMMARY")
    print("=" * 80)

    print(f"\nSkipped (already on Hub): {len(existing_configs)}")
    print(f"Finished successfully: {len(finished)}/{len(missing_configs)}")
    for dataset in finished:
        print(f"  ✓ {dataset}")

    print(f"\nFailed: {len(failed)}/{len(missing_configs)}")
    for item in failed:
        print(f"  ✗ {item['dataset']}")
        print(f"      {item['error']}")


def _dataset_creation_main_sliding_window() -> None:
    """Create every 500 ms sliding cue window from -2500 to +2500 ms."""

    _create_configured_datasets(_sliding_window_dataset_configs())


def _dataset_creation_main() -> None:
    # print(create_dataset())

    """print(create_dataset(negation_target="cue",
                         window_ms=500))
    print(create_dataset(negation_target="scope",
                         window_ms=500))
    print(create_dataset(negation_target="cueandscope",
                         window_ms=500))"""

    """print(create_dataset(negation_target="cue",
                         window_ms=1000))
    print(create_dataset(negation_target="scope",
                         window_ms=1000))
    print(create_dataset(negation_target="cueandscope",
                         window_ms=1000))"""

    """print(create_dataset(negation_target="cue",
                         window_ms_left=2500, window_ms_right=2500))
    print(create_dataset(negation_target="scope",
                         window_ms_left=2500, window_ms_right=2500))
    print(create_dataset(negation_target="cueandscope",
                         window_ms_left=2500, window_ms_right=2500))"""

    """print(create_dataset(negation_target="cue",
                         window_ms_left=100, window_ms_right=100))
    print(create_dataset(negation_target="scope",
                         window_ms_left=100, window_ms_right=100))
    print(create_dataset(negation_target="cueandscope",
                         window_ms_left=100, window_ms_right=100))"""

    # create_dataset_all_words()

    """print(create_dataset(negation_target="cue",
                         event_source="listener",
                         window_ms_left=100, window_ms_right=100))
    print(create_dataset(negation_target="scope",
                         event_source="listener",
                         window_ms_left=100, window_ms_right=100))
    print(create_dataset(negation_target="cueandscope",
                         event_source="listener",
                         window_ms_left=100, window_ms_right=100))

    print(create_dataset(negation_target="cue",
                         event_source="listener",
                         window_ms_left=500, window_ms_right=500))
    print(create_dataset(negation_target="scope",
                         event_source="listener",
                         window_ms_left=500, window_ms_right=500))
    print(create_dataset(negation_target="cueandscope",
                         event_source="listener",
                         window_ms_left=500, window_ms_right=500))"""

    """print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=200, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=300, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=400, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=600, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=700, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=800, window_ms_right=100))
    print(create_dataset(negation_target="cue",
                         event_source="speaker",
                         window_ms_left=900, window_ms_right=100))"""

    _create_configured_datasets(_default_dataset_configs())


if __name__ == "__main__":
    # _dataset_creation_main()
    # _dataset_creation_main_sliding_window()
    """create_dataset(negation_target="cue",
                   event_source="speaker",
                   window_ms_left=1000,
                   window_ms_right=1000,
                   control_mode="very_strict",
                   upload=False)
    create_dataset(
        negation_target="cue",
        event_source="speaker",
        window_ms_left=1000,
        window_ms_right=1000,
        anchor_mode="very_strict",
        control_mode="coarse",
        upload=False,
    )
    create_dataset(
        negation_target="cue",
        event_source="speaker",
        window_ms_left=1000,
        window_ms_right=1000,
        anchor_mode="very_strict",
        control_mode="strict",
        upload=False,
    )"""




    """create_dataset(
        negation_target="cue",
        event_source="speaker",
        window_ms_left=1000,
        window_ms_right=1000,
        anchor_mode="neg",
        control_mode="neg",
        upload=False,
    )"""

    create_dataset(
        negation_target="cue",
        event_source="speaker",
        window_ms_left=1000,
        window_ms_right=1000,
        anchor_mode="neg",
        control_mode="strict",
        upload=False,
    )
