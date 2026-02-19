def set_seed(seed=0):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above
    # NOTE Condider: torch.use_deterministic_algorithms(True)
    # ^ Could be slow
    # TODO Read up on reproducibility in the docs
    # https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
