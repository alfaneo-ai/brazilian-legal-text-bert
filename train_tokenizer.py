import string

import hunspell
import spacy
from gensim.corpora import Dictionary
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer

from tools import *

logger = AppLogger()


class TokenizerTrainer:
    TARGET_MODEL_PATH = 'tokenizer'
    MAX_SEQ_LENGTH = 384
    VOCAB_SIZE = 30000
    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    def __init__(self):
        self.files = [
            PathUtil.build_path('resources', 'corpus_train.txt'),
            PathUtil.build_path('resources', 'corpus_dev.txt')
        ]

    def word_train(self):
        logger.info('Starting tokenizer Word Level training')
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=True,
            lowercase=True
        )
        tokenizer.train(files=self.files, vocab_size=self.VOCAB_SIZE, min_frequency=5, show_progress=True,
                        limit_alphabet=10000, wordpieces_prefix='##', special_tokens=self.SPECIAL_TOKENS)
        tokenizer.save_model(self.TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Word Level trainning')

    def bytes_train(self):
        logger.info('Starting tokenizer Byte Level training')

        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files=self.files, vocab_size=self.VOCAB_SIZE, min_frequency=2, show_progress=True,
                        special_tokens=self.SPECIAL_TOKENS)
        tokenizer.save_model(self.TARGET_MODEL_PATH)
        logger.info('Finished tokenizer Byte Level trainning')


class CreateVocabulary:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg',
                              exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
        self.dictionary = Dictionary([])
        self.spell_checker = SpellChecker()
        self.files = [
            PathUtil.build_path('resources', 'corpus_train.txt'),
            PathUtil.build_path('resources', 'corpus_dev.txt')
            # PathUtil.build_path('resources/bla.txt')
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
        docs = list(self.nlp.pipe(documents, n_process=8))
        tokenized_docs = list()
        for doc in docs:
            tokens = list()
            for token in doc:
                text = self.spell_checker.filter_invalid(token.text)
                if self.__is_valid_token(text) and self.__is_not_null(text):
                    tokens.append(text)
            tokenized_docs.append(tokens)
        return tokenized_docs

    def __create_dictionary(self, documents: list):
        self.dictionary.add_documents(documents)
        output_file = PathUtil.build_path('output', 'dicionario.dict')
        self.dictionary.save_as_text(output_file, sort_by_word=True)

    @staticmethod
    def __is_valid_token(text):
        return text.strip() != '' and text.find('\n') == -1

    @staticmethod
    def __is_not_null(text):
        return text and isinstance(text, str)


class SpellChecker:
    PUNCTUATIONS = r"""́—’‘«­�□δ!"#&'()*+:<=>@[\]^_`{|}~°½º"""

    def __init__(self):
        dict_file = PathUtil.build_path('dictionary', 'pt_PT.dic')
        aff_file = PathUtil.build_path('dictionary', 'pt_PT.aff')
        self.checker = hunspell.HunSpell(dict_file, aff_file)

    def spell(self, token):
        try:
            word = str(token.text)
            typo = self.checker.spell(word)
            if typo is False:
                print(f'{word}')
            return typo
        except TypeError as err:
            print(err)
            return False

    def filter_invalid(self, text):
        return text.translate(str.maketrans('', '', self.PUNCTUATIONS))


if __name__ == '__main__':
    CreateVocabulary().execute()
