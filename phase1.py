# External Lib imports
import collections
import html
import os
import pickle
import re
from functools import partial
from pathlib import Path

import pandas as pd
import sklearn

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, core, lm_rnn

# Torch imports
import torch.nn as nn
import torch.nn.functional as F

# Mytorch imports
from mytorch import loops, lriters
from mytorch.utils.goodies import *

device = torch.device('cuda')
np.random.seed(42)
torch.manual_seed(42)

def compute_mask(t, padding_idx=0):
    """
    compute mask on given tensor t
    :param t:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(t, padding_idx).float()
    return mask


class CustomEncoder(lm_rnn.RNN_Encoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([
            #             torch.nn.ModuleList([self.encoder, self.encoder_with_dropout]),
            torch.nn.ModuleList([self.rnns[0], self.dropouths[0]])
            #             torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
            #             torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])
        ])


class CustomLinear(text.LinearDecoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([self.decoder, self.dropout])


class LanguageModel(nn.Module):

    def __init__(self,
                 _parameter_dict,
                 _device,
                 #                  _wgts_e,
                 #                  _wgts_d,
                 _encargs):
        super(LanguageModel, self).__init__()

        self.parameter_dict = _parameter_dict
        self.device = _device

        self.encoder = CustomEncoder(**_encargs).to(self.device)
        #         self.encoder.load_state_dict(_wgts_e)
        """
            Explanation:
                400*3 because input is [ h_T, maxpool, meanpool ]
                0.4, 0.1 are drops at various layersLM_PATH
        """
        self.linear = CustomLinear(
            _encargs['ntoken'],
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
        score = self.linear(op_p)[0]

        return score

    @property
    def layers(self):
        layers = [x for x in self.encoder.layers]
        layers += [x for x in self.linear.layers]
        return torch.nn.ModuleList(layers)

    @property
    def layers_rev(self):
        layers = [x for x in self.encoder.layers]
        layers += [x for x in self.linear.layers]
        layers.reverse()
        return torch.nn.ModuleList(layers)

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            pred = self.forward(x)
            self.train()
            return pred


class NotSuchABetterEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, debug=False, residual=False):
        '''
            :param max_length: Max length of the sequease do not modify the abstract field unless the comments field is inadequate for your explanation.nce.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
            :param enable_layer_norm: Bool/ layer normalization.
            :param mode: LSTM/GRU.
            :param residual: Bool/ return embedded state of the input.

        TODO: Implement multilayered shit someday.
        '''
        super(NotSuchABetterEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug
        self.mode = mode
        self.residual = residual


        assert self.mode in ['LSTM', 'GRU']

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Mode
        if self.mode == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=1,
                                     bidirectional=self.bidirectional)
        elif self.mode == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    bidirectional=self.bidirectional)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, h):
        """

        :param x: input (batch, seq)
        :param h: hiddenstate (depends on mode. see init hidden)
        :param device: torch device
        :return: depends on booleans passed @ init.
        """

        if self.debug:
            print ("\tx:\t", x.shape)
            if self.mode is "LSTM":
                print ("\th[0]:\t", h[0].shape)
            else:
                print ("\th:\t", h.shape)

        mask = compute_mask(x)

        x = self.embedding_layer(x).transpose(0, 1)

        if self.debug: print ("x_emb:\t\t", x.shape)

        if self.enable_layer_norm:
            seq_len, batch, input_size = x.shape
            x = x.view(-1, input_size)
            x = self.layer_norm(x)
            x = x.view(seq_len, batch, input_size)

        if self.debug: print("x_emb bn:\t", x.shape)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x_sort = x.index_select(1, idx_sort)
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) \
            if self.mode is "LSTM" else h.index_select(1, idx_sort)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort)
        x_dropout = self.dropout.forward(x_pack.data)
        x_pack_dropout = torch.nn.utils.rnn.PackedSequence(x_dropout, x_pack.batch_sizes)

        if self.debug:
            print("\nidx_sort:", idx_sort.shape)
            print("idx_unsort:", idx_unsort.shape)
            print("x_sort:", x_sort.shape)
            if self.mode is "LSTM":
                print ("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print ("h_sort:\t\t", h_sort.shape)


        o_pack_dropout, h_sort = self.rnn.forward(x_pack_dropout, h_sort)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # Unsort o based ont the unsort index we made
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) \
            if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


        # @TODO: Do we also unsort h? Does h not change based on the sort?

        if self.debug:
            if self.mode is "LSTM":
                print("h_sort\t\t", h_sort[0].shape)
            else:
                print("h_sort\t\t", h_sort.shape)
            print("o_unsort\t\t", o_unsort.shape)
            if self.mode is "LSTM":
                print("h_unsort\t\t", h_unsort[0].shape)
            else:
                print("h_unsort\t\t", h_unsort.shape)

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1), h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1), h_unsort, mask


def eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())


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


trn_texts, trn_labels = get_texts_org(IMDB_DATA_PATH / 'train')
val_texts, val_labels = get_texts_org(IMDB_DATA_PATH / 'test')
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

trn_texts, val_texts = trn_texts[:1000], val_texts[:1000]
wiki_trn_texts, wiki_val_texts, wiki_tst_texts = wiki_trn_texts[:len(wiki_trn_texts) // 10], wiki_val_texts[:len(
    wiki_val_texts) // 10], wiki_tst_texts[:len(wiki_tst_texts) // 10]

# Shuffle data
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts, trn_labels = trn_texts[trn_idx], trn_labels[trn_idx]
val_texts, val_labels = val_texts[val_idx], val_labels[val_idx]

# Shuffle data (wiki)
np.random.shuffle(wiki_trn_texts)
np.random.shuffle(wiki_val_texts)
np.random.shuffle(wiki_tst_texts)

wiki_trn_labels = [0 for _ in wiki_trn_texts]
wiki_val_labels = [0 for _ in wiki_val_texts]
wiki_tst_labels = [0 for _ in wiki_val_texts]

chunksize = 24000
re1 = re.compile(r'  +')


def _fixup_(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def _get_texts_(df, n_lbls=1):
    labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)): texts += f' {FLD} {i - n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(_fixup_).values)

    tok = text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))
    return tok, list(labels)


def _simple_apply_fixup_(df):
    labels = [0] * df.shape[0]
    texts = f'\n{BOS} {FLD} 1 ' + df.text
    texts = list(texts.apply(_fixup_).values)
    tok = text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = _get_texts_(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


trn_texts, val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts, val_texts]), test_size=0.1)

if DEBUG:
    print(len(trn_texts), len(val_texts))

df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)

trn_tok, trn_labels = _simple_apply_fixup_(df_trn)
val_tok, val_labels = _simple_apply_fixup_(df_val)

if DEBUG:
    print(f"Trn: {len(trn_tok), len(trn_labels)}, Val: {len(val_tok), len(val_labels)} ")

wiki_trn_texts, wiki_val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([wiki_trn_texts, wiki_val_texts, wiki_tst_texts]), test_size=0.1)

if DEBUG:
    print(len(wiki_trn_texts), len(wiki_val_texts))

wiki_df_trn = pd.DataFrame({'text': wiki_trn_texts, 'labels': [0] * len(wiki_trn_texts)}, columns=col_names)
wiki_df_val = pd.DataFrame({'text': wiki_val_texts, 'labels': [0] * len(wiki_val_texts)}, columns=col_names)

wiki_trn_tok, wiki_trn_labels = _simple_apply_fixup_(wiki_df_trn)
wiki_val_tok, wiki_val_labels = _simple_apply_fixup_(wiki_df_val)

if DEBUG:
    print(f"Trn: {len(wiki_trn_tok), len(wiki_trn_labels)}, Val: {len(wiki_val_tok), len(wiki_val_labels)} ")

'''
    Now we make vocabulary, select 60k most freq words 
        (we do this looking only at imdb, and ignore wiki here)
'''

freq = Counter(p for o in wiki_trn_tok for p in o)
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
em_sz, nh, nl = 400, 256, 1

wd = 1e-7
bptt = 70
bs = 26
opt_fn = partial(torch.optim.Adam, betas=(0.8, 0.99))  # @TODO: find real optimizer, and params

# Load the pre-trained model
# parameter_dict = {'itos2': itos2}
parameter_dict = {}
dps = list(np.asarray([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7)
encargs = {'ntoken': vs,
           'emb_sz': 400, 'n_hid': 256,
           'n_layers': 2, 'pad_token': 0,
           'qrnn': False, 'dropouti': dps[0],
           'wdrop': dps[2], 'dropoute': dps[3], 'dropouth': dps[4]}

# For now, lets assume our best lr = 0.001
bestlr = 0.001 * 5
lm = LanguageModel(parameter_dict, device, encargs)
opt = make_opt(lm, opt_fn, lr=bestlr)

data_fn = partial(text.LanguageModelLoader, bs=bs, bptt=bptt)
data = {'train': np.concatenate(wiki_trn_lm), 'valid': np.concatenate(wiki_val_lm)}
loss_fn = F.cross_entropy

'''
    Schedule

    -> Freeze all but last layer, run for 1 epoch
    -> Unfreeze all of it, and apply discriminative fine-tuning, train normally.
'''
# for grp in opt.param_groups:
#     grp['lr'] = 0.0
# opt.param_groups[-1]['lr'] = 1e-3

lr_args = {'iterations': len(data_fn(data['train'])) * 30}
# lr_args = {'batches':, 'cycles': 1}
# lr_args = {'iterations': len(data_fn(data['train']))*1, 'cut_frac': 0.1, 'ratio': 32}
lr_schedule = lriters.LearningRateScheduler(opt, lr_args, lriters.ConstantLR)

args = {'epochs': 30, 'weight_decay': 0, 'data': data,
        'device': device, 'opt': opt, 'loss_fn': loss_fn, 'train_fn': lm,
        'predict_fn': lm.predict, 'data_fn': data_fn, 'model': lm,
        'eval_fn': eval, 'epoch_start_hook': partial(loops.reset_hidden, lm),
        'clip_grads_at': 0.7, 'lr_schedule': lr_schedule}
traces_start = loops.generic_loop(**args)

MODEL_PATH = Path('resources/proc/wiki')
torch.save(lm.state_dict(), MODEL_PATH / 'pretr_model.torch')
torch.save(lm.encoder.state_dict(), MODEL_PATH / 'pretr_model_enc.torch')
pickle.dump(itos, open(MODEL_PATH / 'itos.pkl', 'wb+'))