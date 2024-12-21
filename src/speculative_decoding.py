import torch
import torch.nn.functional as F
from speculative_sampling import speculative_sample_with_adjusted_distribution
import time

class SpeculativeDecoder:
    def __init__(self, main_model, guide_model, main_tokenizer, guide_tokenizer, max_iterations=3, rejection_threshold=0.5):
        """
        Initialize the SpeculativeDecoder.

        Args:
            main_model: The larger, more accurate language model.
            guide_model: The smaller, faster guiding language model.
            tokenizer: Tokenizer compatible with both models.
            max_iterations: Maximum speculative iterations allowed per token.
            rejection_threshold: Threshold for rejection sampling probability.
        """
        self.main_model = main_model
        self.guide_model = guide_model
        self.tokenizer = main_tokenizer
        #self.main_tokenizer = main_tokenizer
        self.guide_tokenizer = guide_tokenizer
        self.max_iterations = max_iterations
        self.rejection_threshold = rejection_threshold

    def decode_with_timing(self, prompt: str, max_length: int):
        """
        Perform speculative decoding and measure time taken.

        Args:
            prompt: Input string to prime the decoding process.
            max_length: Maximum length of the generated sequence.

        Returns:
            Generated sequence and time taken.
        """
        main_device = next(self.main_model.parameters()).device
        guide_device = next(self.guide_model.parameters()).device

        # Tokenize input prompt with attention mask
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(main_device)
        attention_mask = torch.ones_like(input_ids).to(main_device)

        generated_ids = input_ids.clone()

        start_time = time.time()

        for _ in range(max_length):
            # Generate multiple tokens from guide model
            guide_logits = self.guide_model(
                input_ids=generated_ids.to(guide_device),
                attention_mask=attention_mask.to(guide_device)
            ).logits
            guide_probs = F.softmax(guide_logits[:, -1, :], dim=-1)

            # Perform speculative sampling with adjusted distribution
            adjusted_probs, validated_token = speculative_sample_with_adjusted_distribution(
                guide_probs, self.main_model, generated_ids, main_device, self.max_iterations, self.rejection_threshold
            )

            if validated_token is None:
                # Default to greedy decoding if no validation or rejection probability is None
                main_logits = self.main_model(
                    input_ids=generated_ids.to(main_device),
                    attention_mask=attention_mask.to(main_device)
                ).logits
                main_probs = F.softmax(main_logits[:, -1, :], dim=-1)
                validated_token = torch.argmax(main_probs, dim=-1).item()

            # Append validated token
            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[validated_token]], device=main_device)], dim=-1
            )

            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=main_device)], dim=-1
            )

            # Stop if end-of-sequence token is generated
            if validated_token == self.tokenizer.eos_token_id:
                break

        end_time = time.time()

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True), end_time - start_time
