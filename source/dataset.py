import torch
from itertools import tee
import numpy as np
from random import choice
from pandas import read_csv
from collections import deque

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
                 lookup_labels,
                 length,
                 shuffle,
                 mode):
        self.data_two = data_generator
        if shuffle is True:
            i, entry = DataIterator.__shuffle(self.data_two)
            self.data_one, self.data_two[i] = tee(entry)
        else:
            entry = self.data_two.popleft()
            self.data_one, entry = tee(entry)
            self.data_two.append(entry)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.input_dim = input_dim
        self.lookup_labels = lookup_labels
        self.length = length
        self.shuffle = shuffle
        self.mode = mode

    @staticmethod
    def __shuffle(list_generator):
        entry = choice(list_generator)
        for i in range(len(list_generator)):
            if list_generator[i] is entry:
                return i, choice

    @property
    def size(self):
        return self.length

    @property
    def steps(self):
        return int(self.length / self.batch_size)

    @staticmethod
    def __iter(data, batch_size, lookup_labels, tokenizer, max_len):
        sequences = []
        for i, data in zip(range(batch_size), data):
            if i == batch_size:
                break
            label = np.array([lookup_labels[data[1]]], dtype=np.int64)
            if i == 0:
                labels = label
            else:
                labels = np.concatenate((labels, label), 0)
            sequences.append(data[0])
        return batchify(sequences=sequences, max_len=max_len, tokenizer=tokenizer), torch.from_numpy(labels)

    def __iter__(self):
        try:
            yield DataIterator.__iter(data=self.data_one, batch_size=self.batch_size,
                                      lookup_labels=self.lookup_labels, tokenizer=self.tokenizer, max_len=self.max_len)
        except UnboundLocalError:
            if self.shuffle is True:
                i, entry = DataIterator.__shuffle(self.data_two)
                self.data_one, self.data_two[i] = tee(entry)
            else:
                entry = self.data_two.popleft()
                self.data_one, item = tee(entry)
                self.data_two.append(item)
            yield DataIterator.__iter(data=self.data_one, batch_size=self.batch_size,
                                      lookup_labels=self.lookup_labels, tokenizer=self.tokenizer, max_len=self.max_len)


class Dataset(object):
    def __init__(self,
                 data_path,
                 open_pandas,
                 usecols,
                 tokenizer,
                 batch_size):
        if open_pandas is True:
            self.data = read_csv(data_path, usecols=usecols, chunksize=100 * batch_size)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @staticmethod
    def splits(train_path,
               test_path,
               val_path,
               batch_size,
               usecols,
               tokenizer,
               input_dim,
               max_len,
               lookup_labels,
               shuffle):
        """
            When the original dataset already has been reparteed, this method should be used.
        """
        return DataIterator(data_generator=read_csv(train_path, chunksize=100 * batch_size, usecols=usecols),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=0,
                            shuffle=shuffle,
                            mode='train'), \
               DataIterator(data_generator=read_csv(test_path, chunksize=100 * batch_size, usecols=usecols),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=0,
                            shuffle=False,
                            mode='test'), \
               DataIterator(data_generator=read_csv(val_path, chunksize=100 * batch_size, usecols=usecols),
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=0,
                            shuffle=False,
                            mode='val')

    def split(self, max_len, input_dim, lookup_labels, shuffle, train_test_val=None):
        train = []
        test = deque()
        val = deque()

        for data in self.data:
            data_block = prepare_generator(data)

            train.append(build_generator(data_block[:int(0.7 * len(data_block))]))
            test.append(build_generator(data_block[int(0.7 * len(data_block)):int(0.9 * len(data_block))]))
            val.append(build_generator(data_block[int(0.9 * len(data_block)):]))

        return DataIterator(data_generator=train,
                            tokenizer=self.tokenizer,
                            batch_size=self.batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=len(train),
                            shuffle=shuffle,
                            mode='train'), \
               DataIterator(data_generator=test,
                            tokenizer=self.tokenizer,
                            batch_size=self.batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=len(test),
                            shuffle=False,
                            mode='test'), \
               DataIterator(data_generator=val,
                            tokenizer=self.tokenizer,
                            batch_size=self.batch_size,
                            max_len=max_len,
                            input_dim=input_dim,
                            lookup_labels=lookup_labels,
                            length=len(val),
                            shuffle=False,
                            mode='val')
