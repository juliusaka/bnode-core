def _mixed_norm_tensor(tensor):
    # tensor in torchdiffeq is of shape [batch_size, states_dim]
    mean_per_sample = tensor.abs().pow(2).mean(dim=1).sqrt()
    return mean_per_sample.max()