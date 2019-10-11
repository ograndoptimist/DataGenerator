from source.utils import char
from source.utils import word
from source.utils import build_lookup


class Tokenizer(object):
    def __init__(self,
                 char_level=False,
                 to_lower=True):
        self.vocab = dict()
        self.char_level = char_level
        self.to_lower = to_lower

    def char_iterator(self, text):
        return [char.lower() for word_ in text for char in word_] \
            if self.to_lower else [char_ for word_ in text for char_ in word_]

    def build_vocab(self, text):
        assert len(self.vocab) == 0
        text_vocab = char(text) if self.char_level is True else word(text)
        self.vocab = build_lookup(text_vocab)

    def encode(self, text):
        assert len(self.vocab) > 0
        return [self.vocab[token] for token in self.tokenize(text)]

    def decode(self, list_ids):
        pass

    def tokenize(self, text):
        try:
            if self.char_level is True:
                return self.char_iterator(text)
            else:
                return text.lower().split(" ")
        except AttributeError:
            return text[0].lower().split(" ")

    @property
    def get_vocab(self):
        return self.vocab
