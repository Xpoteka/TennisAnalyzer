"""Stage 7: voice labels (spec section 6.7).

Optional. It transcribes the session audio with word timestamps, keeps the words that are
in ``labels.vocabulary``, and attaches each one to the own hit it comments on: the latest
**confirmed** own contact in the ``labels.max_delay_s`` before the word. Words that map to
nothing, and words with no contact before them, are logged and dropped.

Hand-written labels in ``<labels_dir>/manual_<session>.csv`` (``contact_id,label``)
override the spoken ones for the same contact, so a mis-heard call can be corrected
without re-transcribing.

Output
------
``labels.parquet`` (schema_version 1), one row per label:

* ``swing_id``, ``contact_id`` - the swing the label belongs to
* ``label`` - the vocabulary entry (``good``, ``late``, ...)
* ``word`` - the spoken word it came from, empty for a manual label
* ``t_word`` - when the word was said, on the PTS timeline
* ``t_contact`` - the contact it was attached to
* ``confidence`` - the transcriber's word probability, NaN for a manual label
* ``source`` - ``voice`` or ``manual``

Time
----
Whisper timestamps are relative to ``audio.wav``, and WAV time ``t`` is at PTS
``t + audio_start_s`` (see ARCHITECTURE.md, "Time conventions"). Everything stored here is
already on the PTS timeline.

Transcription backends go through a registry, like the pose backends, so that tests run the
matcher against a synthetic transcript and CI never downloads a model.
"""

from __future__ import annotations

import csv
import logging
import re
import string
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.config import Config, LabelsConfig
from tennis.errors import UserError
from tennis.session import Session
from tennis.util.io import read_json, write_parquet

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("audio.wav", "contacts.parquet", "swing_info.parquet", "metadata.json")
OUTPUTS = ("labels.parquet",)
CONFIG_KEYS = ("labels",)

MODEL_DIR = "models/whisper"
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_NON_WORD = re.compile(r"[^\w']+", re.UNICODE)


@dataclass(frozen=True)
class Word:
    """One transcribed word, in ``audio.wav`` seconds."""

    text: str
    start: float
    end: float
    probability: float = float("nan")


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Path, language: str) -> list[Word]: ...


TranscriberFactory = Callable[[LabelsConfig, Path], Transcriber]
_REGISTRY: dict[str, TranscriberFactory] = {}


def register_transcriber(name: str, factory: TranscriberFactory) -> TranscriberFactory | None:
    previous = _REGISTRY.get(name)
    _REGISTRY[name] = factory
    return previous


def available_transcribers() -> list[str]:
    return sorted(_REGISTRY)


def get_transcriber(config: LabelsConfig, model_root: Path) -> Transcriber:
    factory = _REGISTRY.get(config.backend)
    if factory is None:
        raise UserError(
            f"unknown labels.backend '{config.backend}'; "
            f"available: {', '.join(available_transcribers())}"
        )
    return factory(config, model_root)


class FasterWhisper:
    """faster-whisper with word timestamps. The model is cached under the data root."""

    name = "faster_whisper"

    def __init__(self, config: LabelsConfig, model_root: Path) -> None:
        self.config = config
        self.model_root = model_root

    def transcribe(self, audio: Path, language: str) -> list[Word]:
        from faster_whisper import WhisperModel  # imported late: it loads torch-sized deps

        self.model_root.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            self.config.whisper_model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            download_root=str(self.model_root),
        )
        segments, _ = model.transcribe(str(audio), language=language or None, word_timestamps=True)
        words: list[Word] = []
        for segment in segments:
            for word in segment.words or ():
                words.append(
                    Word(
                        text=str(word.word),
                        start=float(word.start),
                        end=float(word.end),
                        probability=float(getattr(word, "probability", float("nan"))),
                    )
                )
        return words


register_transcriber("faster_whisper", FasterWhisper)


# --- matching ------------------------------------------------------------------------------


def normalize(word: str) -> str:
    """Lower-case, punctuation removed, so ``"Good!"`` and ``good`` are the same word."""
    return _NON_WORD.sub("", word.strip().lower().translate(_PUNCTUATION))


def build_vocabulary(config: LabelsConfig) -> dict[str, str]:
    """Spoken word -> label. Validated in the config, so a word maps to only one label."""
    return {
        normalize(word): label
        for label, words in config.vocabulary.items()
        for word in words
        if normalize(word)
    }


@dataclass(frozen=True)
class LabelRow:
    swing_id: int
    contact_id: int
    label: str
    word: str
    t_word: float
    t_contact: float
    confidence: float
    source: str


@dataclass(frozen=True)
class Contact:
    """A confirmed own hit a label can be attached to."""

    swing_id: int
    contact_id: int
    t_contact: float


def confirmed_contacts(session: Session) -> list[Contact]:
    """Confirmed own hits, in time order (contacts that are not swings cannot be labelled)."""
    rows = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    contacts = [
        Contact(int(r["swing_id"]), int(r["contact_id"]), float(r["t_contact"]))
        for r in rows
        if r.get("is_self_confirmed")
    ]
    return sorted(contacts, key=lambda c: c.t_contact)


def match_words(
    words: Iterable[Word],
    contacts: Sequence[Contact],
    vocabulary: dict[str, str],
    max_delay_s: float,
    audio_start_s: float = 0.0,
) -> tuple[list[LabelRow], list[tuple[str, str]]]:
    """Attach each vocabulary word to the latest confirmed hit just before it.

    Returns the labels and the words that were dropped, each with the reason. A word said
    more than ``max_delay_s`` after the last hit belongs to nothing; a word said before the
    first hit likewise. Several words about the same hit all stick to it - the caller
    decides what to do with that, and the report counts them.
    """
    times = np.array([c.t_contact for c in contacts], dtype=np.float64)
    labels: list[LabelRow] = []
    dropped: list[tuple[str, str]] = []
    for word in words:
        text = normalize(word.text)
        if not text:
            continue
        label = vocabulary.get(text)
        if label is None:
            dropped.append((word.text, "not in the vocabulary"))
            continue
        t_word = float(word.start) + audio_start_s
        i = int(np.searchsorted(times, t_word, side="right")) - 1
        if i < 0:
            dropped.append((word.text, "no own hit before it"))
            continue
        delay = t_word - float(times[i])
        if delay > max_delay_s:
            dropped.append((word.text, f"{delay:.1f}s after the last own hit"))
            continue
        contact = contacts[i]
        labels.append(
            LabelRow(
                swing_id=contact.swing_id,
                contact_id=contact.contact_id,
                label=label,
                word=word.text.strip(),
                t_word=t_word,
                t_contact=contact.t_contact,
                confidence=float(word.probability),
                source="voice",
            )
        )
    return labels, dropped


def manual_labels_path(config: Config, session: Session) -> Path:
    return config.paths.labels_dir / f"manual_{session.id}.csv"


def read_manual_labels(path: Path, contacts: Sequence[Contact]) -> list[LabelRow]:
    """Read ``contact_id,label`` rows. Unknown contact ids are an error, not a silent drop."""
    if not path.is_file():
        return []
    by_contact = {c.contact_id: c for c in contacts}
    with path.open(newline="") as fh:
        rows = [
            r for r in csv.reader(fh) if r and r[0].strip() and not r[0].lstrip().startswith("#")
        ]
    if not rows:
        return []
    if not rows[0][0].strip().lstrip("-").isdigit():
        rows = rows[1:]  # header
    out: list[LabelRow] = []
    for row in rows:
        try:
            contact_id, label = int(row[0]), row[1].strip().lower()
        except (ValueError, IndexError) as exc:
            raise UserError(f"{path}: bad row {','.join(row)!r}: {exc}") from exc
        contact = by_contact.get(contact_id)
        if contact is None:
            raise UserError(
                f"{path}: contact {contact_id} is not a confirmed own hit in this session"
            )
        out.append(
            LabelRow(
                swing_id=contact.swing_id,
                contact_id=contact_id,
                label=label,
                word="",
                t_word=contact.t_contact,
                t_contact=contact.t_contact,
                confidence=float("nan"),
                source="manual",
            )
        )
    return out


def apply_manual(voice: Sequence[LabelRow], manual: Sequence[LabelRow]) -> list[LabelRow]:
    """Manual labels replace every voice label on the same contact (spec section 6.7)."""
    overridden = {r.contact_id for r in manual}
    kept = [r for r in voice if r.contact_id not in overridden]
    return sorted(kept + list(manual), key=lambda r: (r.t_contact, r.t_word, r.label))


SCHEMA = pa.schema(
    [
        ("swing_id", pa.int64()),
        ("contact_id", pa.int64()),
        ("label", pa.string()),
        ("word", pa.string()),
        ("t_word", pa.float64()),
        ("t_contact", pa.float64()),
        ("confidence", pa.float64()),
        ("source", pa.string()),
    ]
)


def labels_table(rows: Sequence[LabelRow]) -> pa.Table:
    return pa.Table.from_pylist([vars(r) for r in rows], schema=SCHEMA)


def read_labels(session: Session) -> list[dict[str, Any]]:
    """``labels.parquet`` as rows, or an empty list when the stage did not run."""
    path = session.path("labels.parquet")
    if not path.exists():
        return []
    return [dict(r) for r in pq.read_table(path).to_pylist()]


def labels_by_swing(session: Session) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in read_labels(session):
        out.setdefault(int(row["swing_id"]), []).append(str(row["label"]))
    return out


def run(ctx: StageContext) -> None:
    session, config = ctx.session, ctx.config
    cfg = config.labels
    meta = read_json(session.path("metadata.json"))
    audio_start = float(meta.get("audio_start_s", 0.0))
    contacts = confirmed_contacts(session)
    vocabulary = build_vocabulary(cfg)

    words: list[Word] = []
    if contacts:
        transcriber = get_transcriber(cfg, config.paths.data_root / MODEL_DIR)
        words = transcriber.transcribe(session.path("audio.wav"), cfg.language)
    else:
        ctx.log("no confirmed own hits; nothing to label", level=logging.WARNING)

    voice, dropped = match_words(words, contacts, vocabulary, cfg.max_delay_s, audio_start)
    manual_path = manual_labels_path(config, session)
    manual = read_manual_labels(manual_path, contacts)
    rows = apply_manual(voice, manual)

    write_parquet(
        labels_table(rows),
        session.path("labels.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config_hash,
        schema_version=SCHEMA_VERSION,
        extra={"language": cfg.language, "whisper_model": cfg.whisper_model},
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    for text, reason in dropped[:20]:
        ctx.log("word dropped", level=logging.DEBUG, word=text, reason=reason)
    ctx.log(
        "labels matched",
        words=len(words),
        labelled=len(rows),
        voice=sum(1 for r in rows if r.source == "voice"),
        manual=len(manual),
        manual_file=str(manual_path) if manual else None,
        dropped=len(dropped),
        swings_labelled=len({r.swing_id for r in rows}),
        counts=counts,
    )
