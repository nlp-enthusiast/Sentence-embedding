import argparse
import os
import gzip
import csv
import utils
import math
import random
from loguru import logger

from models import Transformer,Pooling
from loss import MultipleNegativesRankingLoss

from transformers import AutoTokenizer



def GetArgs():
    parser = argparse.ArgumentParser(description="sentence embedding argparse")

    parser.add_argument("--data_path_prefix", type=str, default="data/")
    parser.add_argument("--data_type", type=str, default="train,dev,test", )
    parser.add_argument("--plm", type=str, help="the pretrained language model path or name", )

    # data hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=75, help="the max length of the model input")

    # train hyper-parameters
    parser.add_argument("--save_dir", type=str, default="./outputs", help="learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="the epoch for training")
    parser.add_argument("--train_batch_size", type=int,
                        default="the number of data samples captured for one training session", )
    parser.add_argument("--dev_batch_size", type=int,
                        default="the number of data samples captured for one training session", )
    args = parser.parse_args()

    return args


def main():
    args = GetArgs()
    model = Transformer(args.plm, max_seq_length=args.max_seq_length)
    pooling=Pooling(model.get_word_embedding_dimension(), pooling_mode='mean')

    nli_dataset_path = 'data/AllNLI.tsv.gz'
    sts_dataset_path = 'data/stsbenchmark.tsv.gz'
    if not os.path.exists(nli_dataset_path):
        utils.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        utils.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Read the AllNLI.tsv.gz file and create the training dataset
    logger.info("Read AllNLI train dataset")

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

    train_data = {}
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()

                add_to_samples(sent1, sent2, row['label'])
                add_to_samples(sent2, sent1, row['label'])  # Also add the opposite

    train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(
                texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(
                texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    logger.info("Train samples: {}".format(len(train_samples)))
    exit()
    # Special data loader that avoid duplicates within a batch
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    # Our training loss
    train_loss = MultipleNegativesRankingLoss(model)

    # Read STSbenchmark dataset and use it as development set
    logger.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                     name='sts-dev')

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs= args.num_epochs,
              evaluation_steps=int(len(train_dataloader) * 0.1),
              warmup_steps=warmup_steps,
              output_path=args.save_path,
              use_amp=False  # Set to True, if your GPU supports FP16 operations
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    model = Transformer(args.save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                      name='sts-test')
    test_evaluator(model, output_path=args.save_path)

if __name__ == '__main__':
    main()
