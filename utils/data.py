import gzip
import csv
import math
import random
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
        return math.floor(len(self.train_examples) / self.batch_size)

class SentenceEmbeddingDataset(Dataset):
    def __init__(self, datapath: str,mode:str="train"):
        super(SentenceEmbeddingDataset, self).__init__()
        self.path = datapath
        self.mode=mode
        if datapath.endswith("tsv"):
            self.data,self.samples = self.load_tsv_data()


    def load_tsv_data(self):
        data = dict()
        samples = []
        with open(self.path, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            if self.mode=="train":
                for row in reader:
                    if row['split'] == 'train':
                        sent1 = row['sentence1'].strip()
                        sent2 = row['sentence2'].strip()
                        label = row['label']
                        if sent1 not in data:
                            data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                        data[sent1][label].add(sent2)
                samples = self.get_train_samples(data)
            elif self.mode=="dev" or self.mode=="test":
                for row in reader:
                    if row['split'] == self.mode :
                        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                        samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            else:
                raise TypeError("you must choose train or dev")
        return data,samples

    def get_train_samples(self,data):
        samples=[]
        for sent1, others in data.items():
            if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
                samples.append(InputExample(
                    texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
                samples.append(InputExample(
                    texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))
        return samples

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
