import math
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader

from tools import *


class SimCseTrain:
    def __init__(self, model_name_or_path):
        self.logger = AppLogger()
        self.downloader = Downloader()
        self.sampler = ExamplePreparer()
        self.model_name = model_name_or_path
        self.train_batch_size = 5
        self.num_epochs = 1
        self.max_seq_length = 256

    def train(self):
        corpus_dataloader = self.prepare_corpus_dataloader()
        dev_evaluator, test_evaluator = self.prepare_sts_evaluators()
        model = self.prepare_model()
        model_saved_path = self.make_train(model, corpus_dataloader, dev_evaluator)
        self.test_model(model_saved_path, test_evaluator)

    def prepare_corpus_dataloader(self):
        corpus_path = 'resources/corpus_samples.txt'
        corpus_samples = self.sampler.prepare_mlm(corpus_path)
        corpus_dataloader = DataLoader(corpus_samples,
                                       shuffle=True,
                                       batch_size=self.train_batch_size,
                                       drop_last=True)
        return corpus_dataloader

    def prepare_sts_evaluators(self):
        dev_path = 'resources/dev.csv'
        test_path = 'resources/test.csv'
        dev_samples = self.sampler.prepare_sts(dev_path)
        test_samples = self.sampler.prepare_sts(test_path)
        dev_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples,
                                                                         batch_size=self.train_batch_size,
                                                                         name='sts-dev')
        test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples,
                                                                          batch_size=self.train_batch_size,
                                                                          name='sts-test')
        return dev_evaluator, test_evaluator

    def prepare_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def make_train(self, model, corpus_dataloader, dev_evaluator):
        dev_evaluator(model)

        model_save_path = 'output/simcse-{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        warmup_steps = math.ceil(len(corpus_dataloader) * self.num_epochs * 0.1)
        evaluation_steps = int(len(corpus_dataloader) * 0.2)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        model.fit(train_objectives=[(corpus_dataloader, train_loss)],
                  evaluator=dev_evaluator,
                  epochs=self.num_epochs,
                  evaluation_steps=evaluation_steps,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  optimizer_params={'lr': 5e-5},
                  save_best_model=True,
                  use_amp=True
                  )
        return model_save_path

    @staticmethod
    def test_model(model_saved_path, test_evaluator):
        saved_model = SentenceTransformer(model_saved_path)
        test_evaluator(saved_model, output_path=model_saved_path)


if __name__ == '__main__':
    # trainner = SimCseTrain('neuralmind/bert-base-portuguese-cased')
    trainner = SimCseTrain('output/mlm_1_epochs')
    trainner.train()
