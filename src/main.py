from speculative_decoding import SpeculativeDecoder
import torch
from utils import load_model_and_tokenizer

def load_models(main_model_name: str, guide_model_name: str, device: str):
    """Load the main and guide models with tokenizers."""
    main_model, main_tokenizer = load_model_and_tokenizer(main_model_name, device)
    guide_model, guide_tokenizer = load_model_and_tokenizer(guide_model_name, device)
    return main_model, guide_model, main_tokenizer, guide_tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_model_name = "gpt2-large"
    guide_model_name = "gpt2-medium"

    # Load models and tokenizers
    main_model, guide_model, tokenizer = load_models(main_model_name, guide_model_name, device)

    # Initialize speculative decoder
    speculative_decoder = SpeculativeDecoder(main_model, guide_model, tokenizer)

    # Generate text with speculative decoding
    prompt = "Once upon a time, in a distant land,"
    max_length = 1
    output_speculative = speculative_decoder.decode(prompt, max_length)

    # Generate text without speculative decoding (greedy decoding)
    output_greedy = tokenizer.decode(main_model.generate(
        tokenizer.encode(prompt, return_tensors="pt").to(device),
        max_length=max_length,
        do_sample=False
    )[0], skip_special_tokens=True)

    # Print outputs
    print("Speculative Decoding Output:")
    print(output_speculative)
    print("\nGreedy Decoding Output:")
    print(output_greedy)

if __name__ == "__main__":
    main()