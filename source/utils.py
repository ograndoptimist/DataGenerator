import pandas as pd


def word(text,
         to_lower=True):
    """
        Responsible to build a word level vocabulary.
    """
    return {word_.lower() for example in text for word_ in example.split(" ")} \
        if to_lower else {word_ for example in text for word_ in example.split(" ")}


def char(text,
         to_lower=True):
    """
        Responsible to build a char level vocabulary.
    """
    return {char_.lower() for example in text for word_ in example for char_ in word_} \
        if to_lower else {char_ for example in text for word_ in example for char_ in word_}


def build_lookup(text):
    """
        Responsible to build a lookup dictionary where each key is a word and
        its respective value is a related id.
    """
    return {word_: i for i, word_ in enumerate(text)}


def prepare_generator(dataframe_generator):
    return [(x, y.split('/')[0]) for x, y in zip(dataframe_generator[dataframe_generator.columns[0]],
                                                 dataframe_generator[dataframe_generator.columns[1]])]


def build_generator(data):
    return (x for x in data)


def save_dataframe(dataframe,
                   data_path,
                   check):
    with open(data_path, 'a') as file:
        if check == 0:
            dataframe.to_csv(file, index=None)
        else:
            dataframe.to_csv(file, index=None, header=None)


def split_dataset(dataset_generator,
                  train_test_split,
                  data_path):
    for i, dataset_chunk in enumerate(dataset_generator):
        train = dataset_chunk[:int(len(dataset_chunk) * train_test_split[0])]

        test = dataset_chunk[int(len(dataset_chunk) * train_test_split[0]): int(
            len(dataset_chunk) * (train_test_split[0] + train_test_split[1]))]

        val = dataset_chunk[int(len(dataset_chunk) * (train_test_split[0] + train_test_split[1])):]

        save_dataframe(train, data_path=data_path + 'train.csv', check=i)
        save_dataframe(test, data_path=data_path + 'test.csv', check=i)
        save_dataframe(val, data_path=data_path + 'val.csv', check=i)


def read_data(data_path, usecols, chunksize):
    return pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)


def generator_word(data_generator):
    return {word_ for data_chunk in data_generator for row in data_chunk[data_chunk.columns[0]] for word_ in
            row.split()}


def generator_char(data_generator):
    return {char_ for data_chunk in data_generator for row in data_chunk[data_chunk.columns[0]] for char_ in row}
