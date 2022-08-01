import math
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from tools import *


def prepare_sts_from_assin2(dataframe: pd.DataFrame):
    samples = []
    for index, row in dataframe.iterrows():
        ementa1 = row['ementa1']
        ementa2 = row['ementa2']
        score = float(row['similarity']) / 5.0
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
        dataloader_first_task = self._prepare_dataloader_first_task()
        evaluator_first_task = self._prepare_evaluator_first_task()
        dataloader_second_task = self._prepare_dataloader_second_task()
        model_to_train = self._prepare_model()
        self._make_train(model_to_train, dataloader_first_task, dataloader_second_task, evaluator_first_task)

    def _prepare_dataloader_first_task(self):
        filepath = os.path.join('resources', self.train_type, 'train.csv')
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        dataloader = DataLoader(samples, shuffle=True, batch_size=self.train_batch_size, drop_last=True)
        return dataloader

    def _prepare_dataloader_second_task(self):
        filepath = os.path.join('resources', 'assin2', 'train.xml')
        dataframe = DatasetManager().from_assin2(filepath)
        samples = prepare_sts_from_assin2(dataframe)
        dataloader = DataLoader(samples, shuffle=True, batch_size=self.train_batch_size, drop_last=True)
        return dataloader

    def _prepare_evaluator_first_task(self):
        filepath = os.path.join('resources', self.train_type, 'dev.csv')
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        return EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                                name=self.train_type, show_progress_bar=True)

    def _prepare_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                   use_auth_token='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')

    def _make_train(self, model_to_train, dataloader_first_task, dataloader_second_task, dev_evaluator):
        checkpoint_path = os.path.join('output', 'checkpoints')
        checkpoint_steps = int(len(dataloader_first_task) * 1)
        saved_path = os.path.join('output', self.train_type)
        warmup_steps = math.ceil(len(dataloader_first_task) * self.num_epochs * 0.1)
        train_loss_first_task = self._get_loss(model_to_train)
        train_loss_second_task = self._get_loss(model_to_train)
        model_to_train.fit(train_objectives=[(dataloader_first_task, train_loss_first_task),
                                             (dataloader_second_task, train_loss_second_task)],
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
