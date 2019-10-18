import torch

from source.tensorizer import batchify

from source.utils import build_generator
from source.utils import prepare_generator


class DataIterator(object):
    def __init__(self,
                 data_generator,
                 tokenizer,
                 batch_size,
                 max_len,
                 input_dim,
                 lookup_labels):
        self.data = data_generator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.input_dim = input_dim
        self.lookup_labels = lookup_labels

    def __iter__(self):
        sequences = []
        for i, data in zip(range(self.batch_size), self.data):
            label = torch.tensor([self.lookup_labels[data[1]]], dtype=torch.float)
            if i == self.batch_size:
                break
            if i == 0:
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
            sequences.append(data[0])
        yield batchify(sequences=sequences, max_len=self.max_len, tokenizer=self.tokenizer), labels


class Dataset(object):
    def __init__(self,
                 data_generator,
                 tokenizer):
        self.data = data_generator
        self.tokenizer = tokenizer

    def split(self, batch_size, max_len, input_dim, lookup_labels):
        train = test = val = []

        for data in self.data:
            data_block = prepare_generator(data)

            [train.append(data) for data in data_block[:int(0.7 * len(data_block))]]
            [test.append(data) for data in data_block[int(0.7 * len(data_block)):int(0.9 * len(data_block))]]
            [val.append(data) for data in data_block[int(0.9 * len(data_block)):]]

        return DataIterator(data_generator=build_generator(train),
                            tokenizer=self.tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels), \
               DataIterator(data_generator=build_generator(test),
                            tokenizer=self.tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels), \
               DataIterator(data_generator=build_generator(val),
                            tokenizer=self.tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels)