# peft-tradeoff-study

A controlled experiment: **same model, same data, same budget, different fine-tuning method.**

Method under test → measured outcome:

| Run | What trains |
|---|---|
| `head_only` | only the classification head (linear probe) |
| `full_ft` | every weight (fp32 master weights, fp16 mixed precision) |
| `lora_r4/16/64` | LoRA adapters on all attention + MLP projections, fp16 frozen base |
| `qlora_r16/64` | same adapters on a 4-bit NF4 quantised base |

For every run it records **test accuracy, macro-F1, peak training VRAM, wall-clock time, throughput, % trainable parameters, saved checkpoint size and inference latency**, then plots the trade-offs.

Default setup: `Qwen/Qwen2.5-0.5B` as a sequence classifier on AG News (4 classes), with 8k train and 2k test examples. Everything is in `configs/study.yaml`.

## Run on Kaggle

Push to `github.com/Sidvortex/peft-tradeoff-study`, import `kaggle_notebook.ipynb`, enable GPU (T4) and Internet, and run all. The full sweep takes about 2–3 h.

`run_all.py` launches every configuration in a **fresh process**, so each peak-VRAM number is clean. It also skips runs already in `results/results.jsonl`, so after a timeout you just re-run it.

```bash
python run_all.py                         # everything
python run_all.py --only lora_r16 qlora_r16
python run_all.py --set data.n_train=2000 # faster, noisier
python report.py                          # table + charts in results/
```

## Outputs

- `results/summary.md`: the full comparison table.
- `results/accuracy_vs_vram.png`: the core trade-off plot.
- `results/accuracy_vs_params.png`: accuracy vs trainable parameters (log scale).
- `results/cost_bars.png`: training time and checkpoint size per run.

## Making it report-worthy

- **Repeat with 3 seeds** (`--set data.seed=1`, then `2`, then `3`, each with its own `train.results_file`) and report mean ± std. Single-seed differences under ~0.5 % are noise.
- **Add a harder task** (a generative one such as GSM8K) where the full-FT vs LoRA gap is known to be larger than on topic classification.
- **Scale the model** (`model.name=Qwen/Qwen2.5-1.5B`). Full FT will OOM on a T4 while QLoRA still fits, which is itself a key finding to show.
- Note that `head_only` saves full weights here, so its checkpoint size is an upper bound. Only the head actually changes.

The expected pattern (verify it, don't assume it): LoRA/QLoRA land within about 1 % of full FT at a fraction of the memory and a ~1000× smaller checkpoint. QLoRA cuts memory further but trains slower because of dequantisation.
