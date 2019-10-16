from source.tensorizer import batchify

from source.utils import build_generator
from source.utils import prepare_generator


class DataIterator(object):
    def __init__(self,
                 data_generator,
                 tokenizer,
                 batch_size,
                 max_len,
                 input_dim):
        self.data = data_generator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.input_dim = input_dim

    def __iter__(self):
        sequences = []
        for i, x in enumerate(self.data):
            if i == self.batch_size:
                break
            sequences.append(x)
        batch = batchify(sequences=sequences, max_len=self.max_len, tokenizer=self.tokenizer)
        yield batch


class Dataset(object):
    def __init__(self,
                 data_generator):
        self.data = data_generator

    def split(self, batch_size, max_len, input_dim, tokenizer):
        train = test = val = []

        for data in self.data:
            data_block = prepare_generator(data)

            [train.append(data) for data in data_block[:int(0.7 * len(data_block))]]
            [test.append(data) for data in data_block[int(0.7 * len(data_block)):int(0.9 * len(data_block))]]
            [val.append(data) for data in data_block[int(0.9 * len(data_block)):]]

        return DataIterator(data_generator=build_generator(train),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim), \
               DataIterator(data_generator=build_generator(test),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim), \
               DataIterator(data_generator=build_generator(val),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim)
