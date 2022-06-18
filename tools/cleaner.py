import re
from copy import deepcopy


class Cleaner:
    def clear(self, paragraphs):
        if not paragraphs:
            return paragraphs
        is_array = isinstance(paragraphs, list)
        texts = deepcopy(paragraphs)
        if not is_array:
            texts = [texts]
        texts = self._remove_undesired_chars(texts)
        texts = self._remove_multiples_spaces(texts)
        texts = self._remove_multiples_dots(texts)
        texts = self._remove_citation(texts)
        texts = self._remove_space_in_last_period(texts)
        texts = self._remove_last_number(texts)
        texts = self._strip_spaces(texts)
        if not is_array:
            texts = texts[0]
        return texts

    @staticmethod
    def _remove_undesired_chars(paragraphs):
        return [re.sub(r'[”“●_\n\t\'\"]', '', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _remove_multiples_spaces(paragraphs):
        return [re.sub(r'\s+', ' ', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _remove_multiples_dots(paragraphs):
        paragraphs = [re.sub(r'\.+', '.', paragraph) for paragraph in paragraphs]
        return [re.sub(r'^\.\s', '', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _remove_citation(paragraphs):
        return [re.sub(r'[\[\(].+[\]\)]', '', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _remove_space_in_last_period(paragraphs):
        return [re.sub(r'\s\.$', '.', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _remove_last_number(paragraphs):
        return [re.sub(r'\.\d+$', '.', paragraph) for paragraph in paragraphs]

    @staticmethod
    def _strip_spaces(paragraphs):
        return [paragraph.strip() for paragraph in paragraphs]
