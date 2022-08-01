import math
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from tools import *


def normalize_ementas(dataframe: pd.DataFrame):
    dataframe['ementa1'] = dataframe.apply(lambda row: text_normalization.normalize(row['ementa1']), axis=1)
    dataframe['ementa2'] = dataframe.apply(lambda row: text_normalization.normalize(row['ementa2']), axis=1)
    return dataframe


def normalize_similarity(dataframe: pd.DataFrame) -> pd.DataFrame:
    similarities = dataframe['similarity'].astype(float).tolist()
    data = np.array(similarities)
    normalized_x = (data - np.min(data)) / (np.max(data) - np.min(data))
    dataframe['similarity'] = normalized_x
    return dataframe


def prepare_sts(dataframe: pd.DataFrame):
    samples = []
    for index, row in dataframe.iterrows():
        ementa1 = row['ementa1'].lower()
        ementa2 = row['ementa2'].lower()
        score = float(row['similarity'])
        samples.append(InputExample(texts=[ementa1, ementa2], label=score))
    samples = shuffle(samples, random_state=0)
    return samples


class MultiTaskStsTrainer:
    def __init__(self, model_name, num_epochs, train_batch_size, max_seq, is_sample=False):
        self.logger = AppLogger()
        self.downloader = Downloader()
        self.sampler = ExamplePreparer()
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.max_seq_length = max_seq
        self.is_sample = is_sample
        self.to_lowercase = True
        self.train_type = 'scale'

    def train(self):
        pesquisa_pronta = self._prepare_pesquisa_pronta_dataloader()
        assin2 = self._prepare_assin2_dataloader()
        evaluator = self._prepare_evaluator()
        model_to_train = self._prepare_model()
        self._make_train(model_to_train, pesquisa_pronta, assin2, evaluator)

    def _prepare_pesquisa_pronta_dataloader(self):
        filepath = os.path.join('resources', 'scale', 'train.csv')
        dataframe = DatasetManager().from_csv(filepath)
        dataframe = normalize_similarity(dataframe)
        dataframe = normalize_ementas(dataframe)
        samples = prepare_sts(dataframe)
        dataloader = DataLoader(samples, shuffle=True, batch_size=self.train_batch_size, drop_last=True)
        return dataloader

    def _prepare_assin2_dataloader(self):
        filepath = os.path.join('resources', 'assin2', 'train.xml')
        dataframe = DatasetManager().from_assin2(filepath)
        dataframe = normalize_similarity(dataframe)
        samples = prepare_sts(dataframe)
        dataloader = DataLoader(samples, shuffle=True, batch_size=self.train_batch_size, drop_last=True)
        return dataloader

    def _prepare_evaluator(self):
        filepath = os.path.join('resources', 'scale', 'dev.csv')
        dataframe = DatasetManager().from_csv(filepath)
        dataframe = normalize_similarity(dataframe)
        dataframe = normalize_ementas(dataframe)
        samples = prepare_sts(dataframe)
        return EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                                name=self.train_type, show_progress_bar=True)

    def _prepare_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                   use_auth_token='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')

    def _make_train(self, model_to_train, pesquisa_pronta, assin2, dev_evaluator):
        checkpoint_path = os.path.join('output', 'checkpoints')
        checkpoint_steps = int(len(pesquisa_pronta) * 1)
        saved_path = os.path.join('output', self.train_type)
        warmup_steps = math.ceil(len(pesquisa_pronta) * self.num_epochs * 0.1)
        pesquisa_pronta_loss = self._get_loss(model_to_train)
        assin2_loss = self._get_loss(model_to_train)
        model_to_train.fit(train_objectives=[(assin2, assin2_loss),
                                             (pesquisa_pronta, pesquisa_pronta_loss)],
                           evaluator=dev_evaluator,
                           epochs=self.num_epochs,
                           warmup_steps=warmup_steps,
                           output_path=saved_path,
                           optimizer_params={'lr': 5e-5},
                           save_best_model=True,
                           checkpoint_path=checkpoint_path,
                           checkpoint_save_steps=checkpoint_steps,
                           use_amp=True)
        message = 'Automatic commit'
        name = self._get_trained_model_name()
        model_to_train.save_to_hub(name, organization='juridics', private=False, commit_message=message,
                                   exist_ok=True, replace_model_card=False)

        return saved_path

    @staticmethod
    def _get_loss(model_to_train):
        return losses.CosineSimilarityLoss(model=model_to_train)

    @staticmethod
    def _get_trained_model_name():
        return 'jurisbert-base-portuguese-multi-sts'


if __name__ == '__main__':
    model, epochs, batch_size, max_seq, train_type, sample, to_lowercase = parse_commands()
    trainner = MultiTaskStsTrainer(model, epochs, batch_size, max_seq, sample)
    trainner.train()
