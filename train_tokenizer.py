import os

import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import BertTokenizer

from tools import *

TARGET_MODEL_PATH = 'tokenizer'
MAX_SEQ_LENGTH = 384
VOCAB_SIZE = 29794
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
logger = AppLogger()


class TokenizerTrainner:
    def __init__(self):
        self.files = [
            os.path.abspath('resources/corpus_train.txt'),
            os.path.abspath('resources/corpus_dev.txt'),
            os.path.abspath('resources/corpus_sts.txt')
        ]

    def word_train(self):
        logger.info('Starting tokenizer Word Level training')
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True
        )
        tokenizer.train(files=self.files, vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=True,
                        limit_alphabet=1000, wordpieces_prefix='##', special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Word Level trainning')

    def bytes_train(self):
        logger.info('Starting tokenizer Byte Level training')

        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files=self.files, vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=True,
                        special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Byte Level trainning')


class SentencePieceTrainner:
    @staticmethod
    def train():
        logger.info('Starting sentencepiece training')
        spm.SentencePieceTrainer.train(input='resources/corpus_train.txt',
                                       model_prefix=TARGET_MODEL_PATH,
                                       vocab_size=VOCAB_SIZE,
                                       user_defined_symbols=SPECIAL_TOKENS)
        logger.info('Finished sentencepiece training')


class TokenizerTester:

    @staticmethod
    def test():
        # path = os.path.abspath(TARGET_MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(TARGET_MODEL_PATH)
        logger.info(tokenizer("Vivendo com otimismo e fé"))


def export_sts_ementas():
    dataset = DatasetManager().from_csv('resources/full.csv')
    sentences1 = dataset['ementa1'].tolist()
    sentences2 = dataset['ementa2'].tolist()
    sentences = list(set(sentences1 + sentences2))
    outfile = open('resources/corpus_sts.txt', 'wb')
    sentences = Cleaner().clear(sentences)
    for sentence in sentences:
        outfile.write(f'{sentence.strip()}\n'.encode())
    outfile.close()


if __name__ == '__main__':
    TokenizerTrainner().word_train()
