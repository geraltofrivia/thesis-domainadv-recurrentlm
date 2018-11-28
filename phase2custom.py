# External Lib imports
import collections
import html
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import sklearn

os.environ['QT_QPA_PLATFORM']='offscreen'

# FastAI Imports
from fastai import text, core, lm_rnn

# Torch imports
import torch.nn as nn
import torch.tensor as T

# Mytorch imports
from mytorch.utils.goodies import *

device = torch.device('cuda')
np.random.seed(42)
torch.manual_seed(42)


class CustomEncoder(lm_rnn.RNN_Encoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([torch.nn.ModuleList([self.rnns[0], self.dropouths[0]]),
                                    torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
                                    torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])])
class CustomLinear(text.LinearDecoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([self.decoder, self.dropout])


class LanguageModel(nn.Module):

    def __init__(self,
                 _device,
                 _wgts_e,
                 _encargs):
        super(LanguageModel, self).__init__()

        self.device = _device

        self.encoder = CustomEncoder(**_encargs).to(self.device)
        self.encoder.load_state_dict(_wgts_e)
        self.decoder = CustomLinear(
            _encargs['ntoken'],
            n_hid=400,
            dropout=0.1 * 0.7,
            tie_encoder=self.encoder.encoder,
            bias=False
        ).to(self.device)

        self.domain = CustomLinear(
            2,
            n_hid=400,
            dropout=0.1 * 0.7,
            tie_encoder=self.encoder.encoder,
            bias=False
        ).to(self.device)
        self.encoder.reset()

    def forward(self, x):
        # Encoding all the data
        op_p = self.encoder(x)

        # pos_batch = op_p[1][-1][-1]
        score = self.decoder(op_p)[0]

        return score

    @property
    def layers(self):
        layers = [x for x in self.encoder.layers]
        layers += [x for x in self.decoder.layers]
        return torch.nn.ModuleList(layers)

    @property
    def layers_rev(self):
        layers = [x for x in self.encoder.layers]
        layers += [x for x in self.decoder.layers]
        layers.reverse()
        return torch.nn.ModuleList(layers)

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            pred = self.forward(x)
            self.train()
            return pred


DEBUG = True

# Path fields
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

WIKI_DATA_PATH = Path('raw/wikitext/wikitext-103/')
WIKI_DATA_PATH.mkdir(exist_ok=True)
IMDB_DATA_PATH = Path('raw/imdb/aclImdb/')
IMDB_DATA_PATH.mkdir(exist_ok=True)
PATH = Path('resources/proc/imdb')
DATA_PROC_PATH = PATH / 'data'
DATA_LM_PATH = PATH / 'datalm'

LM_PATH = Path('resources/models')
LM_PATH.mkdir(exist_ok=True)
PRE_PATH = LM_PATH / 'wt103'
PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'
CLASSES = ['neg', 'pos', 'unsup']
WIKI_CLASSES = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']

def get_texts_org(path):
    texts, labels = [], []
    for idx, label in enumerate(CLASSES):
        for fname in (path / label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts), np.array(labels)

trn_texts, _ = get_texts_org(IMDB_DATA_PATH / 'train')
val_texts, _ = get_texts_org(IMDB_DATA_PATH / 'test')
col_names = ['labels', 'text']
print(len(trn_texts), len(val_texts))

def is_valid_sent(x):
    x = x.strip()
    if len(x) == 0: return False
    if x[0] == '=' and x[-1] == '=': return False
    return True
def wiki_get_texts_org(path):
    texts = []
    for idx, label in enumerate(WIKI_CLASSES):
        with open(path / label, encoding='utf-8') as f:
            texts.append([sent.strip() for sent in f.readlines() if is_valid_sent(sent)])
    return tuple(texts)

wiki_trn_texts, wiki_val_texts, wiki_tst_texts = wiki_get_texts_org(WIKI_DATA_PATH)
print(len(wiki_trn_texts), len(wiki_val_texts))

# Shuffle data
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

# Shuffle data (wiki)
np.random.shuffle(wiki_trn_texts)
np.random.shuffle(wiki_val_texts)
np.random.shuffle(wiki_tst_texts)
"""
    DataFrame Black magic
"""
chunksize = 24000
re1 = re.compile(r'  +')

def _fixup_(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def _simple_apply_fixup_(df):
    labels = [0] * df.shape[0]
    texts = f'\n{BOS} {FLD} 1 ' + df.text
    texts = list(texts.apply(_fixup_).values)
    tok = text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))
    return tok, list(labels)


trn_texts, val_texts = sklearn.model_selection.train_test_split(
        np.concatenate([trn_texts, val_texts]), test_size=0.1)

if DEBUG:
    print(len(trn_texts), len(val_texts))

df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)

trn_tok, trn_labels = _simple_apply_fixup_(df_trn)
val_tok, val_labels = _simple_apply_fixup_(df_val)

if DEBUG:
    print(f"IMDB Trn: {len(trn_tok), len(trn_labels)}, Val: {len(val_tok), len(val_labels)} ")

wiki_trn_texts, wiki_val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([wiki_trn_texts, wiki_val_texts, wiki_tst_texts]), test_size=0.1)

if DEBUG:
    print(len(wiki_trn_texts), len(wiki_val_texts))

wiki_df_trn = pd.DataFrame({'text': wiki_trn_texts, 'labels': [0] * len(wiki_trn_texts)}, columns=col_names)
wiki_df_val = pd.DataFrame({'text': wiki_val_texts, 'labels': [0] * len(wiki_val_texts)}, columns=col_names)

wiki_trn_tok, wiki_trn_labels = _simple_apply_fixup_(wiki_df_trn)
wiki_val_tok, wiki_val_labels = _simple_apply_fixup_(wiki_df_val)

if DEBUG:
    print(f"WIKI Trn: {len(wiki_trn_tok), len(wiki_trn_labels)}, Val: {len(wiki_val_tok), len(wiki_val_labels)} ")

'''
    Now we make vocabulary, select 60k most freq words 
        (we do this looking only at imdb, and ignore wiki here)
'''

freq = Counter(p for o in trn_tok for p in o)
# freq.most_common(25)
max_vocab = 60000
min_freq = 2

itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')
stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
vs = len(itos)

trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

if DEBUG:
    print(f"ITOS: {len(itos)}, STOI: {len(stoi)}")

wiki_trn_lm = np.array([[stoi[o] for o in p] for p in wiki_trn_tok])
wiki_val_lm = np.array([[stoi[o] for o in p] for p in wiki_val_tok])

if DEBUG:
    print(f"ITOS: {len(itos)}, STOI: {len(stoi)}")

"""
    Now we pull pretrained models from disk
"""
em_sz, nh, nl = 400, 1150, 3
# PRE_PATH = PATH / 'models' / 'wt103'
# PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = core.to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)
itos2 = pickle.load((PRE_PATH / 'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i, w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r >= 0 else row_m

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))
wgts_enc = {'.'.join(k.split('.')[1:]): val
            for k, val in wgts.items() if k[0] == '0'}
wgts_dec = {'.'.join(k.split('.')[1:]): val
            for k, val in wgts.items() if k[0] == '1'}


