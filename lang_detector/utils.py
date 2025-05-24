def split_dataset(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]