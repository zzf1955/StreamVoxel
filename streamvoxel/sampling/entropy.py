def entropy_scores(logits):
    """Negative Shannon entropy per voxel (nats, FP32 log-softmax over the class axis). logits [C,Y,Z,X]."""
    logp = logits.float().log_softmax(0)
    return (logp.exp() * logp).sum(0)
