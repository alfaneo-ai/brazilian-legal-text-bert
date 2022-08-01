import re
import unicodedata

from .constants import PUNCTUATIONS


class TextNormalization:
    def __init__(self):
        self.spell = Spell()

    def normalize(self, text):
        cleaned_text = text
        if self.__is_not_null(text):
            cleaned_text = self.__normalize_encoding(cleaned_text)
            cleaned_text = self.__remove_breakline(cleaned_text)
            cleaned_text = self.__remove_consecutive_chars(cleaned_text)
            cleaned_text = self.__fix_spelling(cleaned_text)
            cleaned_text = self.__remove_punctuations(cleaned_text)
            cleaned_text = self.__fix_typos(cleaned_text)
            cleaned_text = self.__to_lower(cleaned_text)
        return cleaned_text

    @staticmethod
    def __is_not_null(text):
        return text and isinstance(text, str)

    @staticmethod
    def __normalize_encoding(text: str):
        return unicodedata.normalize('NFKD', text)

    def __fix_spelling(self, text: str):
        text = self.spell.fix_spelling_ementa(text)
        return text

    @staticmethod
    def __remove_consecutive_chars(text: str):
        text = re.sub(r' +', ' ', text).strip()
        text = re.sub(r'-+', '-', text).strip()
        text = re.sub(r'–+', '-', text).strip()
        text = re.sub(r'\.+', '.', text).strip()
        return text

    @staticmethod
    def __remove_breakline(text: str):
        return re.sub(r'[\r|\n|\r\n]+', '', text)

    @staticmethod
    def __remove_punctuations(text: str):
        return text.translate(str.maketrans('', '', PUNCTUATIONS))

    def __fix_typos(self, text: str):
        text = self.spell.fix_typos(text)
        return text

    @staticmethod
    def __to_lower(text: str):
        return text.lower()


class Spell:
    @staticmethod
    def fix_spelling_ementa(text: str):
        return re.sub(r'[Ee]\s?[Mm]\s?[Ee]\s?[Nn]\s?[Tt]\s?[Aa]:?', 'EMENTA', text,
                      flags=re.RegexFlag.IGNORECASE).strip()

    @staticmethod
    def fix_typos(text: str):
        return re.sub(r"(\s[-–/\-\-\.])(\S+)", r" \g<2>", text, flags=re.IGNORECASE)
