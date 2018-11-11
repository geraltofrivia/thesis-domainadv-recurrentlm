"""
    Pulls an unsupervised fine tuned model from disk, also data, and goes to town on it.
"""

# External Lib imports
import re
import html
import pickle
import sklearn
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import AnyStr, Callable
from sklearn.model_selection import train_test_split

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, core, lm_rnn

# Torch imports
import torch
import torch.nn as nn
import torch.tensor as T
import torch.optim as optim
import torch.nn.functional as F

# Mytorch imports
from mytorch import loops, lriters as mtlr, dataiters as mtdi
from mytorch.utils.goodies import *

device = torch.device('cuda')
np.random.seed(42)
torch.manual_seed(42)

'''
    Paths and macros
'''

DEBUG = True

# Path fields
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

DATA_PATH = Path('raw/imdb/aclImdb/')
DATA_PATH.mkdir(exist_ok=True)
PATH = Path('resources/proc/imdb')
DATA_PROC_PATH = PATH / 'data'
DATA_LM_PATH = PATH / 'datalm'

LM_PATH = Path('resources/models')
LM_PATH.mkdir(exist_ok=True)
PRE_PATH = LM_PATH / 'wt103'
PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'
CLASSES = ['neg', 'pos', 'unsup']

'''
    Model code
'''


class CustomEncoder(lm_rnn.MultiBatchRNN):
    @property
    def layers(self):
        return torch.nn.ModuleList([torch.nn.ModuleList([self.rnns[0], self.dropouths[0]]),
                                    torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
                                    torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])])


class TextClassifier(nn.Module):

    @staticmethod
    def freeze_layer(layer):
        for params in layer.parameters():
            params.requires_grad = False

    @staticmethod
    def unfreeze_layer(layer):
        for params in layer.parameters():
            params.requires_grad = True

    # @TODO: dont take param from param dict, but manually. Also inject comments.
    def __init__(self, _device: torch.device, ntoken: int, dps: list, enc_wgts, _debug=False):
        super(TextClassifier, self).__init__()

        self.device = _device

        # Load the pre-trained model
        args = {'ntoken': ntoken, 'emb_sz': 400, 'n_hid': 1150,
                'n_layers': 3, 'pad_token': 0, 'qrnn': False, 'bptt': 70, 'max_seq': 1400,
                'dropouti': dps[0], 'wdrop': dps[1], 'dropoute': dps[2], 'dropouth': dps[3]}
        self.encoder = CustomEncoder(**args).to(self.device)
        self.encoder.load_state_dict(enc_wgts)
        '''
            Make new classifier.
            
            Explanation:
                400*3 because input is [ h_T, maxpool, meanpool ]
                50 is hidden layer dim
                2 is n_classes

                0.4, 0.1 are drops at various layers
        '''
        self.linear = text.PoolingLinearClassifier(layers=[400 * 3, 50, 2], drops=[dps[4], 0.1]).to(self.device)
        self.encoder.reset()

    def train(self, x, y, loss_fn):
        score = self.forward(x, y)
        loss = loss_fn(score, y)
        return loss, score

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

    def forward(self, x, y):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returrns loss
        '''
        # inputs are S*B

        # Encoding all the data
        op_p = self.encoder(x)
        # pos_batch = op_p[1][-1][-1]
        score = self.linear(op_p)[0]

        return score

    def _eval(self):
        self.encoder.eval()
        self.linear.eval()

    def _train(self):
        self.encoder.train()
        self.linear.train()

    def predict(self, ques):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self._eval()
            op_p = self.encoder(ques)

            predicted = self.linear(op_p)[0]
            self._train()
            return predicted

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder)]

    def load_from(self, location):
        # Pull the data from disk
        if self.debug: print("loading Bilstmdot model from", location)
        self.encoder.load_state_dict(torch.load(location)['encoder'])
        if self.debug: print("model loaded with weights ,", self.get_parameter_sum())


'''
    Prepare data
'''
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = text.Tokenizer().proc_all_mp(core.partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


itos2 = pickle.load((DATA_LM_PATH / 'tmp' / 'itos.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})

chunksize = 24000
df_trn = pd.read_csv(DATA_LM_PATH / 'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(DATA_LM_PATH / 'test.csv', header=None, chunksize=chunksize)

_, trn_labels = get_all(df_trn, 1)
_, val_labels = get_all(df_val, 1)

# trn_labels = np.squeeze(np.load(CLAS_PATH / 'tmp' / 'trn_labels.npy'))
# val_labels = np.squeeze(np.load(CLAS_PATH / 'tmp' / 'val_labels.npy'))

trn_clas = np.load(DATA_LM_PATH / 'tmp' / 'trn_ids.npy')
val_clas = np.load(DATA_LM_PATH / 'tmp' / 'val_ids.npy')

# @TODO: see below
print("OH SHIT CHECK IF DATA MAKES SENSE ACROSS trn_labels and trn_clas!")

'''
    Make model
'''
dps = list(np.asarray([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5)
# enc_wgts = torch.load(LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = torch.load(PATH/'unsup_model_enc.torch', map_location=lambda storage, loc: storage)
clf = TextClassifier(device, len(itos2), dps, enc_wgts)

'''
    Setup things for training (data, loss, opt, lr schedule etc
'''
bs = 24
loss_fn = torch.nn.CrossEntropyLoss()
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
opt = make_opt(clf, opt_fn, lr=0.0)
opt.param_groups[-1]['lr'] = 0.01

# Make data
data_fn = partial(mtdi.SortishSampler, _batchsize=bs)
data = {'train': {'x': trn_clas, 'y': trn_labels}, 'valid': {'x': val_clas, 'y': val_labels}}

# Make lr scheduler
lr_args = {'iterations': len(data_fn(data['train'])), 'cycles': 1}
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)

def epoch_end_hook()-> None:
    lr_schedule.reset()


args = {'epochs': 1, 'data': data, 'device': device,
        'opt': opt, 'loss_fn': loss_fn, 'model': clf,
        'train_fn': clf.train, 'predict_fn': clf.predict,
        'epoch_end_hook': epoch_end_hook, 'weight_decay': 1e-7,
        'clip_grads_at': 0.30, 'lr_schedule': lr_schedule,
        'data_fn': data_fn, 'eval_fn': eval}

'''
    Training schedule:
    
    1. Unfreeze one layer. Train for 1 epoch
    2 - 5. Unfreeze one layer, train for 1 epoch
    3. Train for 15 epochs (after all layers are unfrozen). Use 15 cycles for cosine annealing.
'''
# opt.param_groups[-1]['lr'] = 0.01
loops.generic_loop(**args)

opt.param_groups[-2]['lr'] = 0.001
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)
args['lr_schedule'] = lr_schedule
loops.generic_loop(**args)

opt.param_groups[-3]['lr'] = 0.0001
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)
args['lr_schedule'] = lr_schedule
loops.generic_loop(**args)

opt.param_groups[-4]['lr'] = 0.0001
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)
args['lr_schedule'] = lr_schedule
loops.generic_loop(**args)

opt.param_groups[-5]['lr'] = 0.0001
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)
args['lr_schedule'] = lr_schedule
loops.generic_loop(**args)

lr_args['cycles'] = 15
args['epochs'] = 15
lr_schedule = mtlr.LearningRateScheduler(opt, lr_args, mtlr.CosineAnnealingLR)
args['lr_schedule'] = lr_schedule
loops.generic_loop(**args)



