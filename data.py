"""
    Script which can be called to get data from multiple sources including

        IMDB
        Wikitext103
        Yelp Reviews
        @TODO: What else

    Usage:

    1. Get language model training data for imdb (only text)

    2. Get yelp x & y

    3. Get IMDB data AND then yelp.

"""
import re
import html
import ujson
import pickle
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from fastai import text, core, lm_rnn
from typing import List, Callable, Optional

from mytorch.utils.goodies import *


# Paths & other macros
re1 = re.compile(r'  +')
KNOWN_SOURCES = ['yelp', 'imdb', 'wikitext']
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
IMDB_CLASSES = ['neg', 'pos', 'unsup']
WIKI_CLASSES = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']

WIKI_DATA_PATH = Path('raw/wikitext/wikitext-103/')
WIKI_DATA_PATH.mkdir(exist_ok=True)
IMDB_DATA_PATH = Path('raw/imdb/aclImdb/')
IMDB_DATA_PATH.mkdir(exist_ok=True)


class DataPuller:

    def __init__(self, debug:bool =False, trim_trn:int = 1000, trim_val:int = None,
                 max_vocab: int = 60000, min_freq: int = -1):
        """
            DataPuller object manages pulling data from disk for one or more sources.
            It can pull data both with labels or without.

            Usage:
                1. Make an object with args to suit your needs (see below)
                2. Call .get fn with proper args (multiple times if needed)

        :param debug: bool: if true, will print out lengths and other things once in a while
        :param trim_trn: trim length: if trim flag is enabled during get, will trim the train set acc to specified ints here
            pass negative values or None if you don't want to trim the train set despite the flag
        :param trim_val: same as trim_trn but for validation set.
        :param max_vocab: the max vocab size **for the first dataset**
        :param min_freq: ignore words which occur less than min_freq times while making vocabulary.
        """

        self.debug = debug
        self.max_vocab = max_vocab
        self.min_freq = min_freq

        # Trim lengths
        if trim_trn.__class__ is not int:
            trim_trn = 0
        if trim_val.__class__ is not int:
            trim_val = 0
        self.trim_trn = trim_trn if trim_trn > 0 else None
        self.trim_val = trim_val if trim_val > 0 else None

        self.processed = []
        self.itos, self.stoi = [], {}

    def get(self, src, supervised: bool, trim=False, merge_vocab: int = 0) \
            -> (List[np.ndarray], Optional[List[np.ndarray]], List[np.ndarray], Optional[List[np.ndarray]], List[str]):
        """
            Use this function to pull some dataset from disk

            USAGE:
                @TODO
                data = DataPuller()
                train_imdb, val_imdb = data.get('imdb', labels=True)
                train_yelp, val_yelp = data.get('yelp', labels=True, merge_vocab=0)



        :param src: str: a string depicting which dataset to be pulled.
            Possible values: ['imdb','wikitext','yelp'] @TODO: expand as and when needed
        :param supervised: bool: if True, will return labels, if not, only text.
        :param trim: bool: if True, will only return self.trim_len_trn samples from train,
            and self.trim_len_val samples from validation/test set
        :param merge_vocab: [optional] to be used if multiple datasets are required;
            to indicate whether we want the unique words of the second to be
                case   i: merge_vocab = 0, all replaced by unknown tokens
                case  ii: merge_vocab = n (n > 0) take n most common unique words and give them IDs
                case iii: merge_vocab = n (n < 0) make a new vocab from scratch ignoring previous one.
        :return:
        """

        src = src.lower()
        assert src in KNOWN_SOURCES, f'Incorrect dataset name ({src}) passed.'

        trn_texts, trn_labels, val_texts, val_labels = getattr(self, '_'+src+'_')()

        # Shuffle and Trim (if needed)
        trn_idx = np.random.permutation(len(trn_texts))
        val_idx = np.random.permutation(len(val_texts))

        if trim:
            trn_idx = trn_idx[self.trim_trn]
            val_idx = val_idx[self.trim_val]

        trn_texts, trn_labels = trn_texts[trn_idx], trn_labels[trn_idx]
        val_texts, val_labels = val_texts[val_idx], val_labels[val_idx]

        """
            Vocabulary preparation.
            Two parts - making vocab, and converting the dataset.
            Logic:        
                if this is the first dataset being processed (self.processed is empty),
                    simply make vocab (itos,stoi)
                    and convert this dataset
                    
                if there are other datasets already made, then pass the `merge_vocab` arg to vocab making fn,
                    and correspondingly to the conversion function.
        """
        self._prepare_vocab_(trn_texts, merge_vocab if len(self.processed) > 0 else -1)
        trn_texts, val_texts = self._vocabularize_([trn_texts, val_texts])

        # Finally, return elements depending on `supervised` label
        if supervised:
            return trn_texts, trn_labels, val_texts, val_labels, self.itos.copy()
        else:
            return trn_texts, val_texts, self.itos.copy()

    def _imdb_(self)->(List[str], List[str], List[str], List[str]):
        """ Pulls imdb data from disk, tokenize and return."""
        trn_texts, trn_lbl = self.__imdb_pull_from_disk__(IMDB_DATA_PATH / 'train')
        val_texts, val_lbl = self.__imdb_pull_from_disk__(IMDB_DATA_PATH / 'test')

        if self.debug:
            print(len(trn_texts), len(val_texts))

        col_names = ['labels', 'text']
        df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
        df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)
        trn_tok, trn_lbl = self._apply_fixup_(df_trn)
        val_tok, val_lbl = self._apply_fixup_(df_val)

        # Tokenize text
        trn_tok, val_tok = self.__tokenize__(trn_tok), self.__tokenize__(val_tok)

        return trn_tok, trn_lbl, val_tok, val_lbl

    def _yelp_(self):
        pass

    def _wikitext_(self)->(List[str], List[str], List[str], List[str]):
        """
            Will add validation set to train for our purposes
        :return:
        """
        trn_texts, val_texts, tst_texts = self.__wiki_pull_from_disk__(WIKI_DATA_PATH)

        if self.debug:
            print(len(trn_texts), len(val_texts))

    def _prepare_vocab_(self, source: list, merge_vocab: int = -1)->None:
        """
            Function which takes data (train only), and makes a vocabulary
        :param source: list of tokenized text from which a vocabulary should be made.
        :param merge_vocab: int: if -ve, makes a vocab from scratch,
            if +ve (or 0) will take that many words from current dataset, add to vocab
        :return: None
        """

        freq = Counter(tok for sent in source for tok in sent)

        if merge_vocab < 0:

            # Make new itos from scratch
            itos = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]
            itos.insert(0, '_pad_')
            itos.insert(0, '_unk_')
            stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
            self.itos, self.stoi = itos, stoi

        else:

            tok_sorted = freq.sorted()
            for word, count in tok_sorted:

                if len(self.itos) >= self.max_vocab + merge_vocab:
                    break

                if word not in self.stoi.keys():
                    self.itos.append(word)
                    self.stoi[word] = len(self.stoi)

    def _vocabularize_(self, texts: List[list])->List[np.array]:
        """

        :param texts: Expects a list like [trn_texts, val_texts] where each is a 2D (list of list) object
        :return: list of 2d np arrays
        """

        ids = []
        for toconvert in texts:
            converted = [[self.stoi[o] for o in p] for p in toconvert]
            ids.append(np.array(converted))

        return ids

    @staticmethod
    def __fixup__(x):
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
            ' @-@ ', '-').replace('\\', ' \\ ')
        return re1.sub(' ', html.unescape(x))

    @staticmethod
    def __tokenize__(texts:list):
        """ Uses fastai.text's tokenizer. Expects a simple list."""
        return text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))

    def _apply_fixup_(self, df):
        labels = df.labels.values
        texts = f'\n{BOS} {FLD} 1 ' + df.text
        texts = list(texts.apply(self.__fixup__).values)
        return texts, list(labels)

    @staticmethod
    def __imdb_pull_from_disk__(path):
        texts, labels = [], []
        for idx, label in enumerate(['neg', 'pos', 'unsup']):
            for fname in (path / label).glob('*.*'):
                texts.append(fname.open('r', encoding='utf-8').read())
                labels.append(idx)
        return np.array(texts), np.array(labels)

    @staticmethod
    def __wiki_pull_from_disk__(path):
        texts = []
        for idx, label in enumerate(WIKI_CLASSES):
            with open(path / label, encoding='utf-8') as f:
                texts.append([sent.strip() for sent in f.readlines() if is_valid_sent(sent)])
        return tuple(texts)

    @staticmethod
    def __is_valid_sent__(x):
        x = x.strip()
        if len(x) == 0:
            return False
        if x[0] == '=' and x[-1] == '=':
            return False
        return True
