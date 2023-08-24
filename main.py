import argparse
import os
import gzip
import csv
import math
import random
from loguru import logger

from models import Transformer,Pooling,SentenceTransformer
from loss import MultipleNegativesRankingLoss

import utils
from utils import Dataset,NoDuplicatesDataLoader,EmbeddingSimilarityEvaluator



def GetArgs():
    parser = argparse.ArgumentParser(description="sentence embedding argparse")

    parser.add_argument("--data_path_prefix", type=str, default="data/")
    parser.add_argument("--data_type", type=str, default="train,dev,test", )
    parser.add_argument("--plm", type=str, help="the pretrained language model path or name", )

    # data hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=75, help="the max length of the model input")

    # train hyper-parameters
    parser.add_argument("--save_path", type=str, default="./outputs", help="learning rate")
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
    word_embedding_model = Transformer(args.plm, max_seq_length=args.max_seq_length)
    pooling_model=Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    nli_dataset_path = 'data/AllNLI.tsv'
    sts_dataset_path = 'data/stsbenchmark.tsv'

    logger.info("Read AllNLI train dataset")
    train_dataset=Dataset(nli_dataset_path)
    logger.info("Train samples: {}".format(len(train_dataset)))
    logger.info(train_dataset[0])
    # Special data loader that avoid duplicates within a batch
    train_dataloader = NoDuplicatesDataLoader(train_dataset.samples, batch_size=args.train_batch_size)
    # Our training loss
    train_loss = MultipleNegativesRankingLoss(model)

    # Read STSbenchmark dataset and use it as development set
    logger.info("Read STSbenchmark dev dataset")
    dev_dataset=Dataset(sts_dataset_path,mode="dev")


    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_dataset.samples, batch_size=args.dev_batch_size,
                                                                     name='sts-dev')

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs= args.num_epochs,
              optimizer_params={"lr":args.lr},
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
    test_dataset=Dataset(sts_dataset_path,mode="test")

    model = Transformer(args.save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_dataset.samples, batch_size=args.train_batch_size,
                                                                      name='sts-test')
    test_evaluator(model, output_path=args.save_path)

if __name__ == '__main__':
    main()
