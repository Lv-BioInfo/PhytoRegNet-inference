"""FASTA -> PhytoRegNet -> portable NumPy arrays and a tissue summary."""

import argparse
import csv
import json
import pickle
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .inference import MODEL_NAMES, Predictor, bundled_checkpoint_path
from .sequence import read_fasta


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--fasta", type=Path, help="FASTA with 8192-bp DNA windows."
    )
    inputs.add_argument(
        "--write-example", type=Path,
        help="Write the bundled synthetic example FASTA to this path and exit.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Custom tensor state_dict (.pt); defaults to the bundled model checkpoint.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="arabidopsis")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("outputs/predictions")
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu (default), cuda:0, cuda:1, ..."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads.")
    parser.add_argument(
        "--profile-tsv",
        action="store_true",
        help="Also write one row per sequence/tissue/bin.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing prediction files."
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    if args.write_example is not None:
        path = args.write_example
        if path.exists() and not args.overwrite:
            raise FileExistsError("Example FASTA already exists; add --overwrite to replace it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            files("phytoregnet_minimal").joinpath("data", "demo.fa").read_bytes()
        )
        return [path]
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be positive.")
    prefix = str(args.output_prefix)
    paths = [
        Path(prefix + suffix) for suffix in (".npz", ".counts.tsv", ".metadata.json")
    ]
    if args.profile_tsv:
        paths.append(Path(prefix + ".profile.tsv"))
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(
            "Prediction files already exist; choose a new prefix or add --overwrite."
        )
    checkpoint_path = args.checkpoint or bundled_checkpoint_path(args.model)
    input_paths = {args.fasta.resolve(), checkpoint_path.resolve()}
    if any(path.resolve() in input_paths for path in paths):
        raise ValueError("An output path would overwrite a FASTA or checkpoint input.")
    torch.set_num_threads(args.threads)
    config_name = args.model
    identifiers, sequences = read_fasta(args.fasta)
    predictor = Predictor(config_name, args.checkpoint, args.device)
    values = predictor.predict(sequences, args.batch_size)
    starts = (
        predictor.output_offset
        + np.arange(values["profile"].shape[-1]) * predictor.bin_size
    )
    arrays = {
        **values,
        "sequence_ids": np.asarray(identifiers, dtype=str),
        "track_names": np.asarray(predictor.track_names, dtype=str),
        "track_labels": np.asarray(predictor.track_labels, dtype=str),
        "bin_start0": starts,
        "bin_end0": starts + predictor.bin_size,
        "input_length": np.asarray(predictor.input_length),
        "output_offset": np.asarray(predictor.output_offset),
        "output_length": np.asarray(predictor.output_length),
        "bin_size": np.asarray(predictor.bin_size),
    }
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths[0], **arrays)
    with paths[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sequence_id",
                "tissue",
                "tissue_label",
                "predicted_log_count",
                "predicted_count",
            ]
        )
        for index, identifier in enumerate(identifiers):
            for tissue_index, tissue in enumerate(predictor.track_names):
                writer.writerow(
                    [
                        identifier,
                        tissue,
                        predictor.track_labels[tissue_index],
                        values["log_count"][index, tissue_index],
                        values["count"][index, tissue_index],
                    ]
                )
    metadata = {
        "package_version": __version__,
        "model": args.model,
        "checkpoint_filename": predictor.checkpoint.name,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "input_fasta_filename": args.fasta.name,
        "number_of_sequences": len(identifiers),
        "tracks": predictor.config["tracks"],
        "model_params": predictor.config["model_params"],
        "profile_shape": list(values["profile"].shape),
        "log_count_shape": list(values["log_count"].shape),
        "coordinate_system": "zero-based half-open offsets relative to each FASTA record",
        "output_start0": predictor.output_offset,
        "output_end0": predictor.output_offset + predictor.output_length,
        "profile_scale": "training-normalized signal per bin",
        "count_scale": "max(expm1(raw log_count), 0); training-normalized total signal",
        "checkpoint_count": 1,
        "reverse_complement_averaging": False,
        "dtype": "float32 model inference; float64 count inverse transform",
        "device": str(predictor.device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    paths[2].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.profile_tsv:
        with paths[3].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["sequence_id", "tissue", "bin", "start0", "end0", "predicted_signal"]
            )
            for index, identifier in enumerate(identifiers):
                for tissue_index, tissue in enumerate(predictor.track_names):
                    for bin_index, start in enumerate(starts):
                        writer.writerow(
                            [
                                identifier,
                                tissue,
                                bin_index,
                                start,
                                start + predictor.bin_size,
                                values["profile"][index, tissue_index, bin_index],
                            ]
                        )
    return paths


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        paths = run(args)
    except (
        ValueError,
        OSError,
        RuntimeError,
        FloatingPointError,
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"Wrote {path}")
    return 0
