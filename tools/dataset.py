import os
import xml.etree.ElementTree as ET

import pandas as pd


class DatasetManager:

    @staticmethod
    def from_csv(filepath):
        if not os.path.exists(filepath):
            raise RuntimeError(f'Not found file in {filepath}')
        return pd.read_csv(filepath, sep='|', encoding='utf-8-sig')

    @staticmethod
    def to_csv(dataset, filepath, index=False):
        if os.path.exists(filepath):
            os.remove(filepath)
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(filepath, sep='|', encoding='utf-8-sig', index_label='index', index=index)

    @staticmethod
    def to_file(filepath, texts):
        textfile = open(filepath, 'w')
        if isinstance(texts, list):
            for element in texts:
                textfile.write(element + '\n')
        else:
            textfile.write(texts)
        textfile.close()

    @staticmethod
    def from_text(filepath):
        textfile = open(filepath, 'r')
        text = textfile.read()
        textfile.close()
        return text

    @staticmethod
    def from_assin2(filepath):
        samples = pd.DataFrame({'ementa1': [], 'ementa2': [], 'similarity': []})
        root = ET.parse(filepath).getroot()
        for pair in root.findall('pair'):
            similarity = pair.get('similarity')
            ementa1 = pair.find('h').text
            ementa2 = pair.find('t').text
            item = {
                'ementa1': ementa1,
                'ementa2': ementa2,
                'similarity': similarity
            }
            samples = samples.append(item, ignore_index=True)
        return samples


if __name__ == '__main__':
    DatasetManager().from_xml('../resources/assin2/train.xml')
