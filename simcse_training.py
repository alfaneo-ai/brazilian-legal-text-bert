from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
import logging
from datetime import datetime
import gzip
import sys
import tqdm


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model_name = 'neuralmind/bert-base-portuguese-cased'
train_batch_size = 2
max_seq_length = 384
num_epochs = 1


filepath = '/home/cviegas/Workspace/mestrado/brazilian-legal-text-dataset/output/mlm/corpus.txt'

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = 'output/train_simcse{}-{}'.format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Read the train corpus  #################
train_samples = []
with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10:
            train_samples.append(InputExample(texts=[line, line]))


logging.info("Train sentences: {}".format(len(train_samples)))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 5e-5},
          checkpoint_path=model_output_path,
          show_progress_bar=True,
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )