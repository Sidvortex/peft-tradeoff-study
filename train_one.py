"""Train + measure ONE configuration from configs/study.yaml.

    python train_one.py --run lora_r16
    python train_one.py --run full_ft --set data.n_train=2000      # quicker

Appends one JSON line to results/results.jsonl with accuracy, F1, peak VRAM,
wall-clock time, throughput, trainable params, checkpoint size and latency.
Run each configuration in its own process (run_all.py does this) so VRAM peaks
are measured cleanly.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from src.config import base_parser, load_config
from src.methods import build_model, count_params


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def load_data(dcfg: dict, tok):
    raw = load_dataset(dcfg["dataset"])
    train = raw["train"].shuffle(seed=dcfg["seed"]).select(range(dcfg["n_train"]))
    test = raw["test"].shuffle(seed=dcfg["seed"]).select(range(dcfg["n_test"]))
    label_col = dcfg["label_column"]
    names = raw["train"].features[label_col].names

    def tok_fn(b):
        enc = tok(b[dcfg["text_column"]], truncation=True, max_length=dcfg["max_length"])
        enc["labels"] = b[label_col]
        return enc

    cols = raw["train"].column_names
    return train.map(tok_fn, batched=True, remove_columns=cols), test.map(
        tok_fn, batched=True, remove_columns=cols
    ), names


def main() -> None:
    parser = base_parser(__doc__, "configs/study.yaml")
    parser.add_argument("--run", required=True, help="name of a run in the config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    run = next((r for r in cfg["runs"] if r["name"] == args.run), None)
    if run is None:
        raise SystemExit(f"No run named {args.run!r}. Options: {[r['name'] for r in cfg['runs']]}")
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    torch.manual_seed(dcfg["seed"])

    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds, test_ds, label_names = load_data(dcfg, tok)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(mcfg["name"], len(label_names), tok.pad_token_id, run)
    trainable, total = count_params(model)
    print(f"[{run['name']}] trainable={trainable:,} / total={total:,} ({100 * trainable / total:.3f}%)")

    work = Path(tempfile.mkdtemp(prefix=f"{run['name']}_"))
    targs = TrainingArguments(
        output_dir=str(work),
        num_train_epochs=tcfg["epochs"],
        learning_rate=run["lr"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"] * 4,
        gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_steps=tcfg["warmup_ratio"],  # float < 1 = ratio of total steps
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if run["method"] == "qlora" else "adamw_torch",
        save_strategy="no",
        eval_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=dcfg["seed"],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
    )

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_s = time.perf_counter() - t0
    peak_train_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    t1 = time.perf_counter()
    pred = trainer.predict(test_ds)
    infer_s = time.perf_counter() - t1
    y_pred = np.argmax(pred.predictions, -1)
    y_true = pred.label_ids

    ckpt = work / "final"
    trainer.save_model(str(ckpt))  # adapter-only for (q)lora, full weights otherwise
    ckpt_mb = dir_size_mb(ckpt)

    result = {
        "run": run["name"],
        "method": run["method"],
        "r": run.get("r"),
        "lr": run["lr"],
        "model": mcfg["name"],
        "n_train": dcfg["n_train"],
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "train_loss": round(train_out.training_loss, 4),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "peak_vram_gb": round(peak_train_gb, 2) if peak_train_gb else None,
        "train_minutes": round(train_s / 60, 2),
        "train_samples_per_s": round(dcfg["n_train"] * tcfg["epochs"] / train_s, 1),
        "infer_ms_per_sample": round(1000 * infer_s / len(test_ds), 2),
        "checkpoint_mb": round(ckpt_mb, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(result, indent=2))

    results_file = Path(tcfg["results_file"])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as f:
        f.write(json.dumps(result) + "\n")

    if tcfg.get("keep_checkpoints"):
        dest = Path("checkpoints") / run["name"]
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(ckpt, dest)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
