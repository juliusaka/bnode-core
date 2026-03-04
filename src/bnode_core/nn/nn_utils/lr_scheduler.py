def lr_on_plateau_iterations_to_min_lr(lr_start, lr_min, factor=0.7, eps=1e-5):
    """
    Calculate how many iterations are needed until lr reaches lr_min.
    
    Args:
        lr_start: Starting learning rate
        lr_min: Minimum learning rate threshold
        factor: Decay factor (default 0.7)
        eps: Epsilon value for step adjustment (default 1e-5)
    
    Returns:
        Number of iterations needed to reach lr_min, or None if lr_min is never reached
    """
    def apply_step(lr, factor=0.7, eps=1e-5):
        lr_new = lr * factor
        if lr - lr_new < eps:
            lr_new = lr - eps
        return lr_new

    lr = lr_start
    iterations = 0
    
    # Limit iterations to prevent infinite loops
    max_iterations = 10000
    
    while iterations < max_iterations:
        if lr < lr_min:
            return iterations
        lr = apply_step(lr, factor, eps)
        iterations += 1
    
    return None