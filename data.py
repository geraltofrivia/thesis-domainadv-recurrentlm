"""
    Script which can be called to get data from multiple sources including

        IMDB
        Wikitext103
        Yelp Reviews    [Takes forever, not doing this]

    Usage @TODO:

    1. Get language model training data for imdb (only text)

    2. Get yelp x & y

    3. Get IMDB data AND then yelp.

"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import re
import html
import collections
import pandas as pd
from fastai import text, core
from typing import List, Optional, Union

from mytorch.utils.goodies import *


# Paths & other macros
re1 = re.compile(r'  +')
KNOWN_SOURCES = ['yelp', 'imdb', 'wikitext', 'trec']
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
IMDB_CLASSES = ['neg', 'pos', 'unsup']
WIKI_CLASSES = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']

WIKI_DATA_PATH = Path('./raw/wikitext/wikitext-103/')
IMDB_DATA_PATH = Path('./raw/imdb/aclImdb/')
TREC_DATA_PATH = Path('./raw/trec/')
YELP_DATA_PATH = Path('./raw/yelp/')
CACHED_PATH_TEMPLATE = "./resources/proc/%(src)s/cached"
WIKI_DATA_PATH.mkdir(exist_ok=True)
IMDB_DATA_PATH.mkdir(exist_ok=True)
YELP_DATA_PATH.mkdir(exist_ok=True)


class DataPuller:

    def __init__(self, debug: bool = False, trim_trn: int = 1000, trim_val: int = None,
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

    def get(self, src: str, supervised: bool = True, trim: bool = False, merge_vocab: int = 0, cached: bool = True) \
            -> (List[np.ndarray], Optional[List[np.ndarray]], List[np.ndarray], Optional[List[np.ndarray]], List[str]):
        """
            Use this function to pull some dataset from disk

            USAGE:
                data = DataPuller()
                train_imdb, val_imdb = data.get('imdb', labels=True)
                train_yelp, val_yelp = data.get('yelp', labels=True, merge_vocab=0)


        :param src: str: a string depicting which dataset to be pulled.
            Possible values: ['imdb','wikitext','yelp']
        :param supervised: bool: if True, will return labels, if not, only text.
        :param trim: bool: if True, will only return self.trim_len_trn samples from train,
            and self.trim_len_val samples from validation/test set
        :param merge_vocab: [optional] to be used if multiple datasets are required;
            to indicate whether we want the unique words of the second to be
                case   i: merge_vocab = 0, all replaced by unknown tokens
                case  ii: merge_vocab = n (n > 0) take n most common unique words and give them IDs
                case iii: merge_vocab = n (n < 0) make a new vocab from scratch ignoring previous one.
        :param cached: bool flag which will cache the first dataset requested (and pull it if its the first dataset being pulled with this obj.)
        :return:
        """

        src = src.lower()
        assert src in KNOWN_SOURCES, f'Incorrect dataset name ({src}) passed.'

        # Get data from scratch or cache intelligently
        trn_texts, trn_labels, val_texts, val_labels, from_cache = self._get_(src, merge_vocab=merge_vocab, cached=cached)

        # Shuffle and Trim (if needed)
        trn_idx = np.random.permutation(len(trn_texts))
        val_idx = np.random.permutation(len(val_texts))

        if trim:
            trn_idx = trn_idx[:self.trim_trn]
            val_idx = val_idx[:self.trim_val]

        trn_texts, trn_labels = [trn_texts[i] for i in trn_idx], [trn_labels[i] for i in trn_idx]
        val_texts, val_labels = [val_texts[i] for i in val_idx], [val_labels[i] for i in val_idx]

        if not from_cache:
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

        if not from_cache and not trim:
            # Need to cache it.

            # Cache things if enabled and this is the first dataset we're processing
            if cached and len(self.processed) == 0:

                cache_path = Path(CACHED_PATH_TEMPLATE % {'src': src})
                np.save(cache_path/'trn_texts.npy', trn_texts)
                np.save(cache_path/'trn_labels.npy', trn_labels)
                np.save(cache_path/'val_texts.npy', val_texts)
                np.save(cache_path/'val_labels.npy', val_labels)
                pickle.dump(self.itos, open(cache_path/'itos.pkl', 'wb+'))
                pickle.dump({'trim': trim}, open(cache_path/'options.pkl', 'wb+'))

            # Cache things if enabled and this is the second dataset we're processing
            if cached and len(self.processed) == 1:

                cache_path = Path(CACHED_PATH_TEMPLATE % {'src': self.processed[0]})
                np.save(cache_path / f'trn_texts_{src}.npy', trn_texts)
                np.save(cache_path / f'trn_labels_{src}.npy', trn_labels)
                np.save(cache_path / f'val_texts_{src}.npy', val_texts)
                np.save(cache_path / f'val_labels_{src}.npy', val_labels)
                pickle.dump(self.itos, open(cache_path / f'itos_{src}.pkl', 'wb+'))
                pickle.dump({'trim': trim, 'merge_vocab': merge_vocab}, open(cache_path / f'options_{src}.pkl', 'wb+'))

        # if lists, conv to arrays
        trn_texts = np.array(trn_texts) if type(trn_texts) is list else trn_texts
        trn_labels = np.array(trn_labels) if type(trn_labels) is list else trn_labels
        val_texts = np.array(val_texts) if type(val_texts) is list else val_texts
        val_labels = np.array(val_labels) if type(val_labels) is list else val_labels

        # Finally, return elements depending on `supervised` label | And log what we just accomplished
        self.processed.append(src)

        if supervised:
            return trn_texts, trn_labels, val_texts, val_labels, self.itos.copy()
        else:
            return trn_texts, val_texts, self.itos.copy()

    def _get_(self, src: str, merge_vocab: int = 0, cached: bool = True) \
            -> (List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], bool):
        """ Actually pulls data from cache or disk, vocabularizes etc. """

        # Code to pull processed dataset from disk (first call of get)
        if cached and len(self.processed) == 0:
            try:
                cache_path = Path(CACHED_PATH_TEMPLATE % {'src': src})
                # options = pickle.load(open(cache_path / 'options.pkl', 'rb'))

                # # Check if data was cached in the format that we want right now
                # if options['trim'] != trim:
                #     warnings.warn(f"Data ({src}) is cached but with a different setting than as requested right now.")
                #     raise FileNotFoundError

                # Pulling data from disk
                trn_texts = np.load(cache_path / 'trn_texts.npy')
                trn_labels = np.load(cache_path / 'trn_labels.npy')
                val_texts = np.load(cache_path / 'val_texts.npy')
                val_labels = np.load(cache_path / 'val_labels.npy')
                self.itos = pickle.load(open(cache_path / 'itos.pkl', 'rb'))
                self.stoi = {v: k for k, v in enumerate(self.itos)}

                return trn_texts, trn_labels, val_texts, val_labels, True

            except FileNotFoundError:
                if self.debug:
                    traceback.print_exc()
                warnings.warn(f"Couldn't find requested processed dataset ({src}) from cache, making it from scratch.")

        # Code to pull processed dataset from disk **AFTER ONE** dataset has already been asked for
        if cached and len(self.processed) == 1:
            try:
                cache_path = Path(CACHED_PATH_TEMPLATE % {'src': self.processed[0]})
                options = pickle.load(open(cache_path / f'options_{src}.pkl', 'rb'))

                # Check if data was cached in the format that we want right now
                if options['merge_vocab'] != merge_vocab:
                    warnings.warn(f"Data ({src}) is cached but with a different setting than as requested right now.")
                    raise FileNotFoundError

                # Pulling data from disk
                trn_texts = np.load(cache_path / f'trn_texts_{src}.npy')
                trn_labels = np.load(cache_path / f'trn_labels_{src}.npy')
                val_texts = np.load(cache_path / f'val_texts_{src}.npy')
                val_labels = np.load(cache_path / f'val_labels_{src}.npy')
                self.itos = pickle.load(open(cache_path / f'itos_{src}.pkl', 'rb'))
                self.stoi = {v: k for k, v in enumerate(self.itos)}

                return trn_texts, trn_labels, val_texts, val_labels, True

            except FileNotFoundError:
                if self.debug:
                    traceback.print_exc()
                warnings.warn(f"Couldn't find requested processed dataset ({src}) from cache, making it from scratch.")

        # Cached data not requested or did not exist in proper condition. Making it from scratch
        trn_texts, trn_labels, val_texts, val_labels = getattr(self, '_' + src + '_')()

        return trn_texts, trn_labels, val_texts, val_labels, False

    def _imdb_(self) -> (List[str], List[int], List[str], List[int]):
        """ Pulls imdb data from disk, tokenize and return."""
        trn_texts, trn_lbl = self.__imdb_pull_from_disk__(IMDB_DATA_PATH / 'train')
        val_texts, val_lbl = self.__imdb_pull_from_disk__(IMDB_DATA_PATH / 'test')

        return self.__common_preprocessing_(trn_texts, trn_lbl, val_texts, val_lbl)

    def _trec_(self) -> (List[str], List[int], List[str], List[int]):
        """ Using the 6 class version of the dataset """

        trn_texts, trn_lbl = self.__trec_pull_from_disk__(TREC_DATA_PATH / 'train')
        val_texts, val_lbl = self.__trec_pull_from_disk__(TREC_DATA_PATH / 'test')

        # # One hot encode trn_lbl
        classes = {v: i for i, v in enumerate(np.unique(trn_lbl))}
        trn_lbl = np.array([classes[label] for label in trn_lbl])
        val_lbl = np.array([classes[label] for label in val_lbl])

        return self.__common_preprocessing_(trn_texts, trn_lbl, val_texts, val_lbl)

    def _yelp_(self) -> (List[str], List[int], List[str], List[int]):
        """ Converts 5 star rating into a binary setting (see https://arxiv.org/abs/1801.06146) """
        data = pd.read_json(YELP_DATA_PATH / 'review.json', lines=True)
        texts = data.text.values
        labels = [1 if above else 0 for above in (data.stars > 2).values]

        # To make train, test split (80-20), we set seed at 42 and go to town
        np.random.seed(42)
        idx = np.random.permutation(len(data))
        trn_idx, val_idx = idx[:int(idx.shape[0]*0.8)], idx[int(idx.shape[0]*0.8):]

        trn_texts, trn_lbl = [texts[i] for i in trn_idx], [labels[i] for i in trn_idx]
        val_texts, val_lbl = [texts[i] for i in val_idx], [labels[i] for i in val_idx]

        return self.__common_preprocessing_(trn_texts, trn_lbl, val_texts, val_lbl, _distribute=True)

    def _wikitext_(self) -> (List[str], List[int], List[str], List[int]):
        """ Adds validation set to train """
        trn_texts, val_texts, tst_texts = self.__wiki_pull_from_disk__(WIKI_DATA_PATH)

        if self.debug:
            print(f"Pulled Wikidata from disk with {len(trn_texts)} train, {len(val_texts)} valid and {len(tst_texts)} test samples")

        trn_texts = trn_texts + val_texts

        # For code consistency's sake, we refer to tst as val here on
        #         val_texts = tst_texts

        trn_lbl = [0 for _ in trn_texts]
        tst_lbl = [0 for _ in tst_texts]

        return self.__common_preprocessing_(trn_texts, trn_lbl, tst_texts, tst_lbl)

    def __common_preprocessing_(self, trn_texts: Union[List[str], np.ndarray], trn_lbl: Union[List[int], np.ndarray],
                                val_texts: Union[List[str], np.ndarray], val_lbl: Union[List[int], np.ndarray], _distribute: bool = True) \
            -> (List[str], List[int], List[str], List[int]):
        """ After a point, all datasets need a common preprocessing (fixup etc), we do them here. """
        if self.debug:
            print(len(trn_texts), len(val_texts))

        col_names = ['labels', 'text']
        df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_lbl}, columns=col_names)
        df_val = pd.DataFrame({'text': val_texts, 'labels': val_lbl}, columns=col_names)
        trn_tok, trn_lbl = self._apply_fixup_(df_trn)
        val_tok, val_lbl = self._apply_fixup_(df_val)

        # Tokenize text
        trn_tok, val_tok = self.__tokenize__(trn_tok, _distribute=_distribute), self.__tokenize__(val_tok, _distribute=_distribute)

        return trn_tok, trn_lbl, val_tok, val_lbl

    def _prepare_vocab_(self, source: list, merge_vocab: int = -1) -> None:
        """
            Function which takes data (train only), and makes a vocabulary
        :param source: list of tokenized text from which a vocabulary should be made.
        :param merge_vocab: int: if -ve, makes a vocab from scratch,
            if +ve (or 0) will take that many words from current dataset, add to vocab
        :return: None
        """

        freq = Counter(tok for sent in source for tok in sent)

        if len(self.processed) == 0:

            # Make new itos from scratch
            itos = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]
            itos.insert(0, '_pad_')
            itos.insert(0, '_unk_')
            self.itos, self.stoi = itos.copy(), {v: k for k, v in enumerate(itos)}

        else:

            tok_sorted = freq.sorted()
            vocab_len = len(self.itos)
            for word, count in tok_sorted:

                if len(self.itos) >= vocab_len + merge_vocab:
                    break

                if word not in self.stoi.keys():
                    self.stoi[word] = len(self.itos)
                    self.itos.append(word)

    def _vocabularize_(self, texts: List[list]) -> List[np.array]:
        """

        :param texts: Expects a list like [trn_texts, val_texts] where each is a 2D (list of list) object
        :return: list of 2d np arrays
        """

        ids = []
        for toconvert in texts:
            converted = [[self.stoi.get(o, 0) for o in p] for p in toconvert]
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
    def __tokenize__(texts: list, _distribute: bool = True) -> (List[list]):
        """ Uses fastai.text's tokenizer. Expects a simple list."""
        if _distribute:
            return text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))
        else:
            return text.Tokenizer().proc_all_mp(texts)

    def _apply_fixup_(self, df: pd.DataFrame):
        labels = df.labels.values
        texts = f'\n{BOS} {FLD} 1 ' + df.text
        texts = list(texts.apply(self.__fixup__).values)
        return texts, list(labels)

    @staticmethod
    def __imdb_pull_from_disk__(path: Path) -> (np.ndarray, np.ndarray):
        texts, labels = [], []
        for idx, label in enumerate(['neg', 'pos', 'unsup']):
            for fname in (path / label).glob('*.*'):
                texts.append(fname.open('r', encoding='utf-8').read())
                labels.append(idx)
        return np.array(texts), np.array(labels)

    @staticmethod
    def __trec_pull_from_disk__(path: Path) -> (np.ndarray, np.ndarray):
        # noinspection PyTypeChecker
        raw = open(path, 'r', encoding='ISO-8859-1')
        raw_lbl, raw_txt = [] ,[]
        for line in raw:
            raw_lbl.append(line.split()[0].split(':')[0])
            raw_txt.append(' '.join(line.split()[1:]))
        return np.array(raw_txt), np.array(raw_lbl)

    def __wiki_pull_from_disk__(self, path: Path) -> list:
        texts = []
        for idx, label in enumerate(WIKI_CLASSES):
            # noinspection PyTypeChecker
            with open(path / label, encoding='utf-8') as f:
                texts.append([sent.strip() for sent in f.readlines() if self.__is_valid_sent__(sent)])
        return texts

    @staticmethod
    def __is_valid_sent__(x: str) -> bool:
        x = x.strip()
        if len(x) == 0:
            return False
        if x[0] == '=' and x[-1] == '=':
            return False
        return True


if __name__ == '__main__':

    dd = DataPuller()
    trn_tt, trn_ll, val_tt, val_ll, itos = dd.get('trec', supervised=True)
    print(trn_ll.shape)
