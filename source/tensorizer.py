import numpy as np
import torch


def one_hot(input_dim,
            tokens_vec,
            max_len):
    final_vec = np.zeros((1, input_dim), dtype=np.int64)
    for i, token_id in enumerate(tokens_vec):
        if i == max_len:
            break
        final_vec[0, token_id] = 1
    return final_vec


def batchify(sequences,
             max_len,
             tokenizer):
    batch = np.zeros((len(sequences), len(tokenizer.vocab)), dtype=np.int64)
    for i, text in enumerate(sequences):
        batch[i, :] = one_hot(len(tokenizer.vocab), tokenizer.encode(text), max_len=max_len)
    return torch.from_numpy(batch)
