import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from tools import *

TARGET_MODEL_PATH = 'tokenizer'
MAX_SEQ_LENGTH = 384
VOCAB_SIZE = 29794
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
logger = AppLogger()


class TokenizerTrainner:
    @staticmethod
    def train():
        logger.info('Starting tokenizer training')
        files = [
            os.path.abspath('resources/corpus_train.txt'),
            os.path.abspath('resources/corpus_dev.txt')
        ]
        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files=files, vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=True,
                        special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(TARGET_MODEL_PATH)
        logger.info('Finish tokenizer trainning')


class TokenizerTester:

    @staticmethod
    def test():
        tokenizer = ByteLevelBPETokenizer(
            os.path.abspath(os.path.join(TARGET_MODEL_PATH, 'vocab.json')),
            os.path.abspath(os.path.join(TARGET_MODEL_PATH, 'merges.txt'))
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        logger.info(tokenizer.encode("Vivendo com otimismo e fé").tokens)


if __name__ == '__main__':
    # TokenizerTrainner.train()
    TokenizerTester.test()
