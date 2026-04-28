"""Example: patch Qwen MLP layers with motif-upcycling.

This script is not meant to download weights during tests. Run it in an
internet-enabled environment with Hugging Face access.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from motif_upcycling import MotifUpcycleConfig, SARCConfig, patch_qwen_mlp_layers, trainable_parameter_count

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    cfg = MotifUpcycleConfig(
        num_motifs=6,
        router_hidden=128,
        router_type="contextual",
        lora_rank=8,
        lora_alpha=8.0,
        sarc=SARCConfig(enabled=True, mode="adapter_only", max_delta=0.10),
    )
    patch_qwen_mlp_layers(model, layer_indices=[10, 11], config=cfg)
    print(f"trainable params: {trainable_parameter_count(model):,}")

    prompt = "Explain motif-upcycling in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=48)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
