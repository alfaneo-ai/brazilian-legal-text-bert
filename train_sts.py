import math
import os
import zipfile
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator, TripletEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from tools import *


class FinetunningStsTrain:
    def __init__(self, model_name_or_path, num_epochs, train_batch_size, max_seq_length, train_goal, prefix_name,
                 is_sample, to_lower):
        self.logger = AppLogger()
        self.downloader = Downloader()
        self.sampler = ExamplePreparer()
        self.model_name = model_name_or_path
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.max_seq_length = max_seq_length
        self.train_type = train_goal
        self.is_sample = is_sample
        self.prefix_name = prefix_name
        self.to_lowercase = to_lower

    def train(self):
        train_dataloader = self.prepare_train_dataloader()
        dev_evaluator = self.prepare_evaluator('dev.csv')
        untrained_model = self.prepare_model()
        self.make_train(untrained_model, train_dataloader, dev_evaluator)

    def prepare_train_dataloader(self):
        filepath = os.path.join('resources', self.train_type, 'train.csv')
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        dataloader = DataLoader(samples,
                                shuffle=True,
                                batch_size=self.train_batch_size,
                                drop_last=True)
        return dataloader

    def prepare_evaluator(self, filename):
        filepath = os.path.join('resources', self.train_type, filename)
        samples = self.sampler.prepare_sts(filepath, self.train_type, self.is_sample, self.to_lowercase)
        if self.train_type == 'binary':
            evaluator = BinaryClassificationEvaluator.from_input_examples(samples,
                                                                          show_progress_bar=True,
                                                                          batch_size=self.train_batch_size,
                                                                          name=f'sts-{filename}')
        elif self.train_type == 'scale':
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                                         name=f'sts-{filename}', show_progress_bar=True)

        elif self.train_type == 'triplet':
            evaluator = TripletEvaluator.from_input_examples(samples, batch_size=self.train_batch_size,
                                                             name=f'sts-{filename}', show_progress_bar=True)

        return evaluator

    def prepare_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                   use_auth_token='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')

    def make_train(self, model, corpus_dataloader, dev_evaluator):
        model_checkpoint_path = 'output/checkpoints'
        model_save_path = f'output/{self.prefix_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        warmup_steps = math.ceil(len(corpus_dataloader) * self.num_epochs * 0.1)
        evaluation_steps = int(len(corpus_dataloader) * 1)
        train_loss = self.get_loss(model)
        model.fit(train_objectives=[(corpus_dataloader, train_loss)],
                  evaluator=dev_evaluator,
                  epochs=self.num_epochs,
                  # evaluation_steps=evaluation_steps,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  optimizer_params={'lr': 5e-5},
                  save_best_model=True,
                  checkpoint_path=model_checkpoint_path,
                  checkpoint_save_steps=evaluation_steps,
                  use_amp=True
                  )

        message = f'{self.train_type}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        name = f'{self.prefix_name}-sts-{self.train_type}'
        model.save_to_hub(name, organization='juridics', private=False, commit_message=message,
                          exist_ok=True, replace_model_card=False)

        return model_save_path

    def get_loss(self, model):
        if self.train_type == 'binary':
            train_loss = losses.SoftmaxLoss(model=model,
                                            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                            num_labels=2)
        elif self.train_type == 'scale':
            train_loss = losses.CosineSimilarityLoss(model=model)
        elif self.train_type == 'triplet':
            train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE)

        return train_loss


def unzip():
    target = PathUtil.build_path('resources', 'binary')
    source = PathUtil.join(target, 'data.zip')
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(target)


if __name__ == '__main__':
    unzip()
    model, epochs, batch_size, max_seq, train_type, hub_prefix_name, sample, to_lowercase = parse_commands()
    trainner = FinetunningStsTrain(model, epochs, batch_size, max_seq, train_type, hub_prefix_name, sample,
                                   to_lowercase)
    trainner.train()
