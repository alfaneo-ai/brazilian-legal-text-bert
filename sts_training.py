import logging
import math
import os
from datetime import datetime

import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

output_path = 'output/'


class Logger:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
                            handlers=[LoggingHandler()])
        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message)


class ModelDownloader:

    @staticmethod
    def download_model(output):
        modelpath = os.path.join(output_path, output)
        filepath = os.path.join(modelpath, 'model.zip')
        gdd.download_file_from_google_drive(file_id='1j1zaEVU4uO9ukVRHp2iqVN1xPWncmKfR',
                                            dest_path=filepath,
                                            unzip=True)
        return modelpath


class DatasetReader:
    DATASET_ROOTPATH = 'resources/'

    def __init__(self, logger):
        self.logger = logger

    def read(self, filename):
        filepath = os.path.join(self.DATASET_ROOTPATH, filename)
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        self.logger.info(f'Dataset read from {filepath}')
        return dataset


class DatasetPreparer:

    @staticmethod
    def prepare_dataset(dataset, two_way=False):
        result = []
        for index, row in dataset.iterrows():
            result.append(InputExample(texts=[row['ementa1'], row['ementa2']], label=int(row['similarity'])))
            if two_way:
                result.append(InputExample(texts=[row['ementa2'], row['ementa1']], label=int(row['similarity'])))
        return result


class Trainner:
    def __init__(self, logger, batch_size=4, epochs=1):
        self.logger = logger
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, train_dataset, dev_dataset, model_filepath):
        self.logger.info('Starting model trainning')
        checkpoint = 'juridics-legal-bert-sms-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_savepath = os.path.join(output_path, checkpoint)
        model = CrossEncoder(model_filepath, num_labels=1, max_length=128)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_dataset, name='juridics')
        warmup_steps = math.ceil(len(train_dataloader) * self.epochs * 0.1)
        self.logger.info("Warmup-steps: {}".format(warmup_steps))
        model.fit(train_dataloader=train_dataloader,
                  evaluator=evaluator,
                  epochs=self.epochs,
                  evaluation_steps=5000,
                  show_progress_bar=True,
                  warmup_steps=warmup_steps,
                  output_path=model_savepath)


if __name__ == '__main__':
    main_logger = Logger()

    # model_path = ModelDownloader().download_model('juridics-legal-bert')
    model_path = 'neuralmind/bert-base-portuguese-cased'

    reader = DatasetReader(main_logger)
    train_data = reader.read('consultas-prontas-train.csv')
    dev_data = reader.read('consultas-prontas-dev.csv')

    preparer = DatasetPreparer()
    train_samples = preparer.prepare_dataset(train_data, two_way=True)
    dev_samples = preparer.prepare_dataset(dev_data, two_way=False)

    trainner = Trainner(main_logger, batch_size=24, epochs=6)
    trainner.train(train_samples, dev_samples, model_path)
