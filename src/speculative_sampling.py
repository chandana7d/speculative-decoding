import torch
import torch.nn.functional as F

def speculative_sample_with_adjusted_distribution(guide_probs, main_model, generated_ids, device, max_iterations, rejection_threshold):
    """
    Perform sampling based on the condition q(x) ≤ p(x), reject with probability (1 - p(x)/q(x)),
    and resample from adjusted distribution if needed.

    Args:
        guide_probs: Probabilities from the guide model.
        main_model: Main language model for validation.
        generated_ids: Tokens generated so far.
        device: Device to run the model.
        max_iterations: Maximum iterations for sampling.
        rejection_threshold: Threshold for rejection sampling probability.

    Returns:
        Adjusted probabilities and validated token.
    """
    # Ensure guide_probs is normalized and non-negative
    guide_probs = torch.clamp(guide_probs, min=0)
    guide_probs /= guide_probs.sum(dim=-1, keepdim=True)

    samples = torch.multinomial(guide_probs, num_samples=max_iterations)

    # Compute probabilities from main model
    main_logits = main_model(generated_ids.to(device))[0]
    main_probs = F.softmax(main_logits[:, -1, :], dim=-1)

    for sample in samples.squeeze(0):
        if guide_probs[0, sample] <= main_probs[0, sample]:
            return guide_probs, sample.item()
        else:
            rejection_prob = 1 - (main_probs[0, sample] / guide_probs[0, sample])
            if rejection_prob is None or (rejection_prob <= rejection_threshold and torch.rand(1).item() > rejection_prob):
                return guide_probs, sample.item()

    # Adjust the distribution if no valid sample is found
    adjusted_probs = torch.clamp(main_probs - guide_probs, min=0)
    adjusted_probs /= adjusted_probs.sum()
    resampled = torch.multinomial(adjusted_probs, num_samples=1)
    return adjusted_probs, resampled.item()
