import torch


class Tensorizer(object):
    def __init__(self,
                 input_dim):
        self.input_dim = input_dim

    def one_hot(self, tokens_vec):
        final_vec = torch.zeros((len(tokens_vec), self.input_dim))  # [batch_size, input_dim]
        for cont, token in enumerate(tokens_vec):
            final_vec[cont, token] = 1
        return final_vec

    def to_tensor(self, tokens_vec):
        return self.one_hot(tokens_vec)
