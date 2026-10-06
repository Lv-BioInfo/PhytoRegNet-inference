"""Read fixed-length FASTA records and encode A/C/G/T/N."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np


def validate_sequence(sequence: str, length: int, identifier: str = "sequence") -> str:
    sequence = sequence.upper()
    if len(sequence) != length:
        raise ValueError(
            f"{identifier!r} has {len(sequence)} bases; expected exactly {length}. "
            "Supply a fixed-length window; sequences are not padded or cropped."
        )
    invalid = sorted(set(sequence) - set("ACGTN"))
    if invalid:
        raise ValueError(
            f"{identifier!r} contains unsupported bases {invalid}; use A/C/G/T/N."
        )
    return sequence


def read_fasta(path: str | Path, length: int = 8192) -> tuple[list[str], list[str]]:
    """Read multiline FASTA; keep the first header token as the unique ID."""
    identifiers, sequences, seen = [], [], set()
    identifier, pieces = None, []

    def append_record():
        if identifier is None:
            return
        if identifier in seen:
            raise ValueError(f"Duplicate FASTA identifier: {identifier!r}.")
        sequences.append(validate_sequence("".join(pieces), length, identifier))
        identifiers.append(identifier)
        seen.add(identifier)

    with Path(path).open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header on line {line_number}.")
                identifier, pieces = header.split()[0], []
            else:
                if identifier is None:
                    raise ValueError(
                        f"Sequence before the first FASTA header on line {line_number}."
                    )
                pieces.append(line)
    append_record()
    if not sequences:
        raise ValueError("The FASTA file contains no sequences.")
    return identifiers, sequences


def one_hot(sequences: Sequence[str], length: int = 8192) -> np.ndarray:
    """Return float32 [batch, 4, length], channels A/C/G/T; N is all zero."""
    result = np.zeros((len(sequences), 4, length), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        sequence = validate_sequence(sequence, length, f"sequence {index}")
        letters = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        for channel, base in enumerate(b"ACGT"):
            result[index, channel, letters == base] = 1
    return result
