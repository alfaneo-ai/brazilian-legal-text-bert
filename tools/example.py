import pandas as pd
from sentence_transformers.readers import InputExample


class ExamplePreparer:

    @staticmethod
    def prepare_sts(filepath) -> list:
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        result = []
        for index, row in dataset.iterrows():
            result.append(InputExample(texts=[row['ementa1'], row['ementa2']], label=int(row['similarity'])))
            result.append(InputExample(texts=[row['ementa2'], row['ementa1']], label=int(row['similarity'])))
        return result

    @staticmethod
    def prepare_mlm(filepath: str) -> list:
        result = []
        with open(filepath, 'r', encoding='utf8') as inputstream:
            for line in inputstream:
                line = line.strip()
                tokens = line.split()
                if len(tokens) > 10:
                    result.append(InputExample(texts=[line, line]))
        return result
