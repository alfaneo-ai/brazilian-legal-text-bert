import os

import spacy
from gensim.corpora import Dictionary
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer

from tools import *

TARGET_MODEL_PATH = 'tokenizer'
MAX_SEQ_LENGTH = 384
VOCAB_SIZE = 30000
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
logger = AppLogger()


class TokenizerTrainner:
    def __init__(self):
        self.files = [
            os.path.abspath('resources/corpus_train.txt'),
            os.path.abspath('resources/corpus_dev.txt')
        ]

    def word_train(self):
        logger.info('Starting tokenizer Word Level training')
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=True,
            lowercase=True
        )
        tokenizer.train(files=self.files, vocab_size=VOCAB_SIZE, min_frequency=5, show_progress=True,
                        limit_alphabet=10000, wordpieces_prefix='##', special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Word Level trainning')

    def bytes_train(self):
        logger.info('Starting tokenizer Byte Level training')

        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files=self.files, vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=True,
                        special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Byte Level trainning')


class CreateVocabulary:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg',
                              exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
        self.dictionary = Dictionary([])
        self.files = [
            # PathUtil.build_path('resources/corpus_train.txt'),
            PathUtil.build_path('resources/corpus_dev.txt')
        ]

    def execute(self):
        documents = self.__read_documents()
        tokenized_documents = self.__spacy_tokenizer(documents)
        self.__create_dictionary(tokenized_documents)

    def __read_documents(self) -> list:
        documents = list()
        for filepath in self.files:
            with open(filepath, mode='r', encoding='utf-8') as fileinput:
                lines = fileinput.readlines()
                for line in lines:
                    documents.append(line)
        return documents

    def __spacy_tokenizer(self, documents: list) -> list:
        docs = list(self.nlp.pipe(documents, n_process=-1))
        tokenized_docs = list()
        for doc in docs:
            tokens = list()
            for token in doc:
                if self.__is_valid_token(token):
                    tokens.append(token.text)
            tokenized_docs.append(tokens)
        return tokenized_docs

    def __create_dictionary(self, documents: list):
        self.dictionary.add_documents(documents)
        output_file = PathUtil.build_path('outputs', 'dicionario.dict')
        self.dictionary.save_as_text(output_file, sort_by_word=True)

    @staticmethod
    def __is_valid_token(token):
        return token.is_stop is False \
               and token.is_punct is False \
               and token.text.strip() != '' \
               and token.text.find('\n') == -1


if __name__ == '__main__':
    CreateVocabulary().execute()
