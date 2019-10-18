import torch


def one_hot(input_dim,
            tokens_vec,
            max_len):
    final_vec = torch.zeros((max_len, input_dim), dtype=torch.long)  # [batch_size, input_dim]
    for i, token_id in enumerate(tokens_vec):
        if i == max_len:
            break
        final_vec[i, token_id] = 1
    final_vec = final_vec.unsqueeze(0)
    return final_vec


def batchify(sequences,
             max_len,
             tokenizer):
    for cont, text in enumerate(sequences):
        tensor = one_hot(len(tokenizer.vocab), tokenizer.encode(text), max_len=max_len)
        if cont == 0:
            batch = tensor
        else:
            batch = torch.cat((batch, tensor), 0)
    return batch
