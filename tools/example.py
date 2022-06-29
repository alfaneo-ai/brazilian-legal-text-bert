import os
from glob import glob

import pandas as pd
from sentence_transformers.readers import InputExample


def get_files(root_path, extension='*.*'):
    """
        - root_path: Path raiz a partir de onde serão realizadas a busca
        - extension: Extensão de arquivo usado para filtrar o retorno
        - retorna: Retorna todos os arquivos recursivamente a partir de um path raiz
     """
    return [y for x in os.walk(root_path) for y in glob(os.path.join(x[0], extension))]


class ExamplePreparer:

    @staticmethod
    def prepare_sts(filepath, train_type) -> list:
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        dataset = dataset.sample(frac=0.05)
        result = []
        for index, row in dataset.iterrows():
            if train_type == 'binary':
                score = int(row['similarity'])
            elif train_type == 'embedding':
                score = float(row['similarity']) / 4.0
            result.append(InputExample(texts=[row['ementa1'], row['ementa2']], label=score))
            result.append(InputExample(texts=[row['ementa2'], row['ementa1']], label=score))
        return result

    @staticmethod
    def prepare_mlm(rootpath: str) -> list:
        result = []
        filepaths = get_files(rootpath)
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf8') as inputstream:
                for line in inputstream:
                    line = line.strip()
                    tokens = line.split()
                    if len(tokens) > 10:
                        result.append(InputExample(texts=[line, line]))
        return result
