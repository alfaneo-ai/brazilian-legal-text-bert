import math
import os
import zipfile
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator, TripletEvaluator, LabelAccuracyEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader

from tools import *


def unzip():
    target = PathUtil.build_path('resources', 'binary')
    source = PathUtil.join(target, 'data.zip')
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(target)


class StsTrainer(ABC):
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
        self.train_type = None

    @abstractmethod
    def prepare_evaluator(self, filename):
        pass

    @abstractmethod
    def get_loss(self, model_to_train):
        pass

    def train(self):
        dataloader = self._prepare_dataloader()
        evaluator = self.prepare_evaluator('dev.csv')
        model_to_train = self._prepare_model()
        self._make_train(model_to_train, dataloader, evaluator)

    def _prepare_dataloader(self):
        filepath = os.path.join('resources', self.train_type, 'train.csv')
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        dataloader = DataLoader(samples, shuffle=True, batch_size=self.train_batch_size, drop_last=True)
        return dataloader

    def _prepare_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                   use_auth_token='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')

    def _make_train(self, model_to_train, corpus_dataloader, dev_evaluator):
        checkpoint_path = os.path.join('output', 'checkpoints')
        checkpoint_steps = int(len(corpus_dataloader) * 1)
        saved_path = os.path.join('output', self.train_type)
        warmup_steps = math.ceil(len(corpus_dataloader) * self.num_epochs * 0.1)
        train_loss = self.get_loss(model_to_train)
        if dev_evaluator:
            model_to_train.fit(train_objectives=[(corpus_dataloader, train_loss)],
                               evaluator=dev_evaluator,
                               epochs=self.num_epochs,
                               warmup_steps=warmup_steps,
                               output_path=saved_path,
                               optimizer_params={'lr': 5e-5},
                               save_best_model=True,
                               checkpoint_path=checkpoint_path,
                               checkpoint_save_steps=checkpoint_steps,
                               use_amp=True)
        else:
            model_to_train.fit(train_objectives=[(corpus_dataloader, train_loss)],
                               epochs=self.num_epochs,
                               warmup_steps=warmup_steps,
                               output_path=saved_path,
                               optimizer_params={'lr': 5e-5},
                               save_best_model=False,
                               checkpoint_path=checkpoint_path,
                               checkpoint_save_steps=checkpoint_steps,
                               use_amp=True)
        message = 'Automatic commit'
        name = self._get_trained_model_name()
        model_to_train.save_to_hub(name, organization='juridics', private=False, commit_message=message,
                                   exist_ok=True, replace_model_card=False)

        return saved_path

    def _get_trained_model_name(self):
        prefix = '-'.join(self.model_name.split('-')[:2])
        return f'{prefix}-sts-{self.train_type}'


class ContrastiveStsTrainer(StsTrainer):
    def __init__(self, name, num_epochs, batch_size, max_seq, is_sample=False):
        StsTrainer.__init__(self, name, num_epochs, batch_size, max_seq, is_sample)
        self.train_type = 'binary'

    def prepare_evaluator(self, filename):
        filepath = os.path.join('resources', self.train_type, filename)
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        return BinaryClassificationEvaluator.from_input_examples(samples,
                                                                 show_progress_bar=True,
                                                                 batch_size=self.train_batch_size,
                                                                 name=self.train_type)

    def get_loss(self, model_to_train):
        return losses.ContrastiveLoss(model=model_to_train)


class BinaryStsTrainer(StsTrainer):
    def __init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample=False):
        StsTrainer.__init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample)
        self.train_type = 'binary'

    def prepare_evaluator(self, filename):
        filepath = os.path.join('resources', self.train_type, filename)
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        return BinaryClassificationEvaluator.from_input_examples(samples,
                                                                 show_progress_bar=True,
                                                                 batch_size=self.train_batch_size,
                                                                 name=self.train_type)

    def get_loss(self, model_to_train):
        return losses.SoftmaxLoss(model=model_to_train,
                                  sentence_embedding_dimension=model_to_train.get_sentence_embedding_dimension(),
                                  num_labels=2)


class ScaleStsTrainer(StsTrainer):
    def __init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample=False):
        StsTrainer.__init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample)
        self.train_type = 'scale'

    def prepare_evaluator(self, filename):
        filepath = os.path.join('resources', self.train_type, filename)
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        return EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                                name=self.train_type, show_progress_bar=True)

    def get_loss(self, model_to_train):
        return losses.CosineSimilarityLoss(model=model_to_train)


class TripletStsTrainer(StsTrainer):
    def __init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample=False):
        StsTrainer.__init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample)
        self.train_type = 'triplet'

    def prepare_evaluator(self, filename):
        filepath = os.path.join('resources', self.train_type, filename)
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        return TripletEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                    name=self.train_type, show_progress_bar=True)

    def get_loss(self, model_to_train):
        return losses.TripletLoss(model=model_to_train, distance_metric=losses.TripletDistanceMetric.COSINE)


class BatchTripletStsTrainer(StsTrainer):
    def __init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample=False):
        StsTrainer.__init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq, is_sample)
        self.train_type = 'batch_triplet'

    def prepare_evaluator(self, filename):
        return None

    def get_loss(self, model_to_train):
        return losses.BatchAllTripletLoss(model=model_to_train,
                                          distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance)


if __name__ == '__main__':
    unzip()
    model, epochs, batch_size, max_seq, train_type, sample, to_lowercase = parse_commands()
    if train_type == 'binary':
        trainner = BinaryStsTrainer(model, epochs, batch_size, max_seq, sample)
    elif train_type == 'scale':
        trainner = ScaleStsTrainer(model, epochs, batch_size, max_seq, sample)
    elif train_type == 'triplet':
        trainner = TripletStsTrainer(model, epochs, batch_size, max_seq, sample)
    elif train_type == 'contrastive':
        trainner = ContrastiveStsTrainer(model, epochs, batch_size, max_seq, sample)
    elif train_type == 'batch_triplet':
        trainner = BatchTripletStsTrainer(model, epochs, batch_size, max_seq, sample)
    trainner.train()
