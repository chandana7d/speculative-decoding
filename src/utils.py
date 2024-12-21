import torch
import torch.nn.functional as F
from typing import List

def load_model_and_tokenizer(model_name: str, device: str):
    """Load a Hugging Face model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    return model, tokenizer