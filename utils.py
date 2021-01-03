import numpy as np

class Tokenizer(object):
    def __init__(self, 
                 chars='abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ',
                 unk_token=True):
        self.chars = chars
        self.unk_token = 69 if unk_token else None

        self.build()

    def build(self):
        """Build up char2idx.
        """
        self.idx = 1    # idx 0 reserved for zero padding
        self.char2idx = {}
        self.idx2char = {}

        for char in self.chars:
            self.char2idx[char] = self.idx
            self.idx2char[self.idx] = char
            self.idx += 1

    def char_to_idx(self, 
                    c):
        """Return the integer character index of a character token.
        """
        if not c in self.char2idx:
            if self.unk_token is None:
                return None   # Return None if no unknown word's defined
            else:
                return self.unk_token

        return self.char2idx[c]

    def idx_to_char(self, 
                    idx):
        """Return the character string of an integer word index.
        """
        # Unknown token
        if idx > len(self.idx2char):
            if self.unk_token is None:
                return ''
            else:
                return '<UNK>'

        # Return nothing for zero padding
        elif idx == 0:
            return ''
        
        return self.idx2char[idx]

    def __len__(self):
        """Return the length of the vocabulary.
        """
        return len(self.char2idx)

    def text_to_sequence(self, 
                         text,
                         maxlen=1014):
        text = text.lower() # Forced lower casing, as specified in VDCNN paper

        data = np.zeros(maxlen, ).astype(int)
        for i in range(len(text)):
            if i > maxlen:
                return data
            if text[i] in self.char2idx:
                data[i] = self.char_to_idx(text[i])
        return data

    def sequence_to_text(self,
                         seq):
        text = ''
        for idx in seq:
            text += self.idx_to_char(idx)
        return text