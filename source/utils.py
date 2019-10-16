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
    return [data for data in dataframe_generator[dataframe_generator.columns[0]]]


def build_generator(data):
    return (x for x in data)
