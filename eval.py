import argparse
import os
import gzip
import csv
import math
import random
from loguru import logger

from models import Transformer, Pooling, SentenceTransformer
from loss import MultipleNegativesRankingLoss

import utils
from utils import SentenceEmbeddingDataset,TextRankingDataset,NoDuplicatesDataLoader, EmbeddingSimilarityEvaluator,DataLoader


def GetArgs():
    parser = argparse.ArgumentParser(description="sentence embedding argparse")
    parser.add_argument("--queries_path", type=str, default="")
    parser.add_argument("--corpus_path", type=str, default="")
    parser.add_argument("--ce_scores_path", type=str, default="")
    parser.add_argument("--hard_negatives_path", type=str, default="")

    parser.add_argument("--test_data_path", type=str, default="data/test.txt")
    parser.add_argument("--model_name", type=str, help="the pretrained language model path or name", )


    # data hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=75, help="the max length of the model input")


    parser.add_argument("--dev_batch_size", type=int,
                        default="the number of data samples captured for one training session", )

    parser.add_argument("--save_path", type=str, default="./outputs", help="learning rate")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--negs_to_use", default=None,
                        help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--use_all_queries", default=False, action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = GetArgs()
    print(args)

    test_data_name = args.test_data_path.split("/")[-1]

    logger.info(f"Loading Data")

    test_dataset = TextRankingDataset(args, mode="test")
    # test_dataset = SentenceEmbeddingDataset(args, mode="test")
    logger.info(f"test dataset: {test_data_name}  nums:{len(test_dataset)}")

    logger.info("test example:\n"+str(test_dataset[0]))


    model = SentenceTransformer(args.model_name)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_dataset.samples,
                                                                          batch_size=args.dev_batch_size,
                                                                          name='sts-test')
    test_evaluator(model, output_path=args.save_path)



if __name__ == '__main__':
    main()
