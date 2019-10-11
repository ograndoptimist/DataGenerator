import torch
import numpy as np


class Tensorizer(object):
    def __init__(self,
                 input_dim):
        self.input_dim = input_dim

    def to_numpy(self, tokens_vec):
        final_vec = np.zeros((len(tokens_vec), self.input_dim))  # [batch_size, input_dim]
        for token in tokens_vec:
            final_vec[token] = 1
        return final_vec

    def to_tensor(self, tokens_vec):
        numpy_vec = self.to_numpy(tokens_vec)
        tensor = torch.tensor(numpy_vec)
        return tensor
