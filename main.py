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
    parser.add_argument("--train_data_path", type=str, default="data/train.txt")
    parser.add_argument("--queries_path", type=str, default="")
    parser.add_argument("--corpus_path", type=str, default="")
    parser.add_argument("--ce_scores_path", type=str, default="")
    parser.add_argument("--hard_negatives_path", type=str, default="")

    parser.add_argument("--dev_data_path", type=str, default="data/dev.txt")
    parser.add_argument("--test_data_path", type=str, default="data/test.txt")
    parser.add_argument("--model_name", type=str, help="the pretrained language model path or name", )
    parser.add_argument("--dev", action="store_true", default=False, help="是否进行验证", )
    parser.add_argument("--test", action="store_true", default=False, help="是否进行测试", )

    # data hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=75, help="the max length of the model input")

    # train hyper-parameters
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="the epoch for training")
    parser.add_argument("--train_batch_size", type=int,
                        default="the number of data samples captured for one training session", )
    parser.add_argument("--dev_batch_size", type=int,
                        default="the number of data samples captured for one training session", )

    parser.add_argument("--save_path", type=str, default="./outputs", help="learning rate")
    parser.add_argument("--use_amp", action="store_true", default=False, help=" Set to True, if your GPU supports FP16 operations")
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--negs_to_use", default=None,
                        help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--pooling", type=str, default="mean", help="learning rate")
    args = parser.parse_args()

    return args


def main():
    args = GetArgs()
    print(args)

    if args.use_pre_trained_model:
        logger.info("use pretrained model")
        model = SentenceTransformer(args.model_name)
    else:
        logger.info("create new sbert model")
        word_embedding_model = Transformer(args.model_name, max_seq_length=args.max_seq_length)
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data_name = args.train_data_path.split("/")[-1]
    dev_data_name = args.dev_data_path.split("/")[-1]
    test_data_name = args.test_data_path.split("/")[-1]

    logger.info(f"Loading Data")
    train_dataset = TextRankingDataset(args, mode="train")
    # train_dataset = SentenceEmbeddingDataset(args, mode="train")
    train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=args.train_batch_size)
    logger.info(f"train dataset: {train_data_name}  nums:{len(train_dataset)}")

    if args.dev:
        dev_dataset = TextRankingDataset(args, mode="dev")
        # dev_dataset = SentenceEmbeddingDataset(args, mode="dev")
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_dataset.samples,
                                                                         batch_size=args.dev_batch_size,
                                                                         name='sts-dev')
        logger.info(f"dev dataset: {dev_data_name}  nums:{len(dev_dataset)}")
    if args.test:
        test_dataset = TextRankingDataset(args, mode="test")
        # test_dataset = SentenceEmbeddingDataset(args, mode="test")
        logger.info(f"test dataset: {test_data_name}  nums:{len(test_dataset)}")


    logger.info("train example:\n"+str(train_dataset[0]))
    # Our training loss
    train_loss = MultipleNegativesRankingLoss(model)


    # # Configure the training
    # warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  if args.warmup_steps is None else args.warmup_steps # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(args.warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator if args.dev else None,
              epochs= args.num_epochs,
              optimizer_params={"lr":args.lr},
              evaluation_steps=int(len(train_dataloader) * 0.1),
              warmup_steps=args.warmup_steps,
              checkpoint_path=args.save_path,
              use_amp=args.use_amp  # Set to True, if your GPU supports FP16 operations
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    if args.test:
        model = SentenceTransformer(args.save_path)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_dataset.samples,
                                                                          batch_size=args.train_batch_size,
                                                                          name='sts-test')
        test_evaluator(model, output_path=args.save_path)
    # Save the model
    model.save(args.save_path)


if __name__ == '__main__':
    main()
