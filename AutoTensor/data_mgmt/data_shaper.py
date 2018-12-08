def normalize(data):
    """
    Alters all data to be normalized as z-scores, with each column independent
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std