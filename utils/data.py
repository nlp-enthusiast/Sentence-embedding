import gzip
import csv
import json
import math
import pickle
import random
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset


class NoDuplicatesDataLoader:

    def __init__(self, train_examples, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        random.shuffle(self.train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                # 判断是否有重复样本 如果有出现过的句子就跳过
                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        # 下舍整数
        return math.floor(len(self.train_examples) / self.batch_size)


class SentenceEmbeddingDataset(Dataset):
    def __init__(self, args: str, mode: str = "train"):
        super(SentenceEmbeddingDataset, self).__init__()
        if mode == "train":
            self.path = args.train_data_path
        elif mode == "dev":
            self.path = args.dev_data_path
        elif mode == "test":
            self.path = args.test_data_path
        else:
            raise ValueError("mode must in train dev test")
        self.mode = mode
        assert self.path.endswith("tsv"), "only support load .tsv data"
        self.data, self.samples = self.load_tsv_data()

    def load_tsv_data(self):
        '''
        构造三元组 矛盾 蕴含 中性
        :return:
        '''
        data = dict()
        samples = []
        with open(self.path, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            if self.mode == "train":
                for row in reader:
                    if row['split'] == 'train':
                        sent1 = row['sentence1'].strip()
                        sent2 = row['sentence2'].strip()
                        label = row['label']
                        if sent1 not in data:
                            data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                        data[sent1][label].add(sent2)
                samples = self.get_train_samples(data)
            elif self.mode == "dev" or self.mode == "test":
                for row in reader:
                    if row['split'] == self.mode:
                        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                        samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            else:
                raise TypeError("you must choose train or dev")
        return data, samples

    def get_train_samples(self, data):
        samples = []
        for sent1, others in data.items():
            if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
                samples.append(InputExample(
                    texts=[sent1, random.choice(list(others['entailment'])),
                           random.choice(list(others['contradiction']))]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class TextRankingDataset(Dataset):
    def __init__(self, args, mode: str = "train"):
        super(TextRankingDataset, self).__init__()
        self.mode=mode
        if mode=="train":
            self.queries_path = args.train_data_path
        elif mode=="dev":
            self.queries_path = args.dev_data_path
        elif mode=="test":
            self.queries_path = args.test_data_path
        else:
            raise ValueError("mode must in train dev test")
        self.corpus_path = args.corpus_path
        self.ce_scores_path = args.ce_scores_path
        self.hard_negatives_path = args.hard_negatives_path
        self.mode = mode
        self.ce_scores = self.load_ce_scores()
        self.corpus = self.load_corpus()
        print(self.corpus[786450])
        exit()
        self.queries = self.load_queries()
        self.train_queries,max_score,min_score = self.load_hard_negatives(args)
        self.train_queries_ids = list(self.train_queries.keys())

        for qid in self.train_queries:
            self.train_queries[qid]['pos'] = list(self.train_queries[qid]['pos'])
            self.train_queries[qid]['neg'] = list(self.train_queries[qid]['neg'])
            random.shuffle(self.train_queries[qid]['neg'])

        self.samples= []
        for item in range(len(self.train_queries_ids)):
            query = self.train_queries[self.train_queries_ids[item]]
            query_text = query['query']

            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
            if self.mode=="train":
                neg_id = query['neg'].pop(0)  # Pop negative and add at end
                neg_text = self.corpus[neg_id]
                query['neg'].append(neg_id)
                self.samples.append(InputExample(texts=[query_text, pos_text, neg_text]))
            else:
                score= (query["score"]-min_score)/(max_score-min_score)
                self.samples.append(InputExample(texts=[query_text, pos_text, score]))

    def load_corpus(self):
        corpus = dict()
        with open(self.corpus_path, 'r', encoding='utf8') as f:
            for line in f:
                pid, passage = line.strip().split("\t")
                pid = int(pid)
                corpus[pid] = passage
        return corpus

    def load_queries(self):
        queries = dict()
        with open(self.queries_path, 'r', encoding='utf8') as f:
            for line in f:
                qid, query = line.strip().split("\t")
                qid = int(qid)
                queries[qid] = query
        return queries

    def load_ce_scores(self):
        with open(self.ce_scores_path,"rb") as f:
            ce_scores = pickle.load(f)
        return ce_scores

    def load_hard_negatives(self,args):
        train_queries = dict()
        max_score=0
        min_score=1000
        with open(self.hard_negatives_path, 'rt') as f:
            for line in tqdm(f):
                data = json.loads(line)

                # Get the positive passage ids
                qid = data['qid']
                pos_pids = data['pos']

                if len(pos_pids) == 0:  # Skip entries without positives passages
                    continue

                pos_min_ce_score = min([self.ce_scores[qid][pid] for pid in data['pos']])
                pos_max_ce_score = max([self.ce_scores[qid][pid] for pid in data['pos']])
                if self.mode!="train":
                    if data['qid'] not in self.queries:
                        continue
                    print("yes")
                    max_score=max(max_score,pos_max_ce_score)
                    min_score=min(min_score,pos_min_ce_score)
                    train_queries[data['qid']] = {'qid': data['qid'], 'query': self.queries[data['qid']],'pos': pos_pids,"score":pos_min_ce_score}
                ce_score_threshold = pos_min_ce_score - args.ce_score_margin

                # Get the hard negatives
                neg_pids = set()
                if args.negs_to_use is not None:  # Use specific system for negatives
                    negs_to_use = args.negs_to_use.split(",")
                else:  # Use all systems
                    negs_to_use = list(data['neg'].keys())
                # logger.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

                for system_name in negs_to_use:
                    if system_name not in data['neg']:
                        continue

                    system_negs = data['neg'][system_name]
                    negs_added = 0
                    for pid in system_negs:
                        if self.ce_scores[qid][pid] > ce_score_threshold:
                            continue

                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= args.num_negs_per_system:
                                break

                if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                    train_queries[data['qid']] = {'qid': data['qid'], 'query': self.queries[data['qid']], 'pos': pos_pids,
                                             'neg': neg_pids}

        del self.ce_scores
        return train_queries,max_score,min_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):

        return self.samples[item]


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', texts=None, label=0):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

