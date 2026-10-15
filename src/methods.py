"""Build the model for each fine-tuning method being compared."""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

METHODS = ("head", "full", "lora", "qlora")

# Attention + MLP projections of Qwen / Llama / Mistral style models.
# Listed explicitly so the classification head ("score") is never wrapped by LoRA.
LLAMA_STYLE_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_model(name: str, num_labels: int, pad_token_id: int, run: dict):
    method = run["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method}")

    kwargs = {"num_labels": num_labels, "pad_token_id": pad_token_id}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    if method == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["dtype"] = torch.float16
    elif method == "lora":
        # frozen base in fp16 saves memory; trainable adapters are kept in fp32 by PEFT
        kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        # full / head: master weights must be fp32 for stable fp16 mixed precision training
        kwargs["dtype"] = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    model.config.pad_token_id = pad_token_id

    if method == "head":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("score")
        return model

    if method == "full":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        return model

    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=run["r"],
        lora_alpha=run.get("alpha", 2 * run["r"]),
        lora_dropout=run.get("dropout", 0.05),
        target_modules=run.get("target_modules", LLAMA_STYLE_LINEARS),
        modules_to_save=["score"],  # the classification head is trained fully
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora)
    # the head copy made by modules_to_save must be fp32 for fp16 AMP
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model
