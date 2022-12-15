from typing import *

class Preprocessor:
    def __call__(self, profiles):
        raise NotImplementedError

    def encode(self, profile):
        raise NotImplementedError
    def encode_batch(self, profiles): 
        raise NotImplementedError
