# External Lib imports
import os
import re
import html
import argparse
import collections
from pathlib import Path
from typing import Callable
from functools import partial

import pandas as pd
import sklearn
from tqdm import tqdm

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, core, lm_rnn

# Torch imports
import torch.nn as nn
import torch.tensor as T
import torch.nn.functional as F

# Mytorch imports
from mytorch import loops as mtlp
from mytorch.utils.goodies import *
from mytorch import lriters as mtlr

# Local imports
import utils
from data import DataPuller
from options import Phase2 as params

DEVICE = 'cuda:2'


device = torch.device(DEVICE)
np.random.seed(42)
torch.manual_seed(42)

# Path fields
PATH = Path('resources/proc/imdb')
DATA_PROC_PATH = PATH / 'data'
DATA_LM_PATH = PATH / 'datalm'

LM_PATH = Path('resources/models')
LM_PATH.mkdir(exist_ok=True)
PRE_PATH = LM_PATH / 'wt103'
PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'

'''
    Data sampler for this training
'''


class DomainAgnosticSampler:
    """ Sample data for language model training from two different domains in one batch. """

    def __init__(self, data_fn, data_a, data_b):
        """
            Here, data_fn would be something like
                `partial(text.LanguageModelLoader, bs=bs, bptt=bptt)`
            And data_a/b would be something like
                `{'train': np.concatenate(trn_lm), 'valid': np.concatenate(val_lm)}['train']`
            data_fn (fastai's language model loader) flattens y and returns x of seqlen, batchsize
        """
        self.args = {'data_fn': data_fn, 'data_a': data_a, 'data_b': data_b}
        self.reset(**self.args)

    def reset(self, data_fn, data_a, data_b):
        self.itera = iter(data_fn(data_a))
        self.iterb = iter(data_fn(data_b))

    def __iter__(self):
        return self

    def __next__(self):
        x_a, y_a = self.itera.__next__()
        x_b, y_b = self.iterb.__next__()
        return self._combine_batch_(x_a, x_b, y_a, y_b)

    def __len__(self):
        return min(len(self.args['data_fn'](self.args['data_a'])),
                   len(self.args['data_fn'](self.args['data_b'])))

    @staticmethod
    def _combine_batch_(x_a, x_b, y_a, y_b):
        """
            :param x_a is a np.arr looks like seqlen, batchsize
            :param y_a is a corresponding np.arr (one word ahead than x_a) which is a flattened x_a.shape mat
             Same for x_b, y_b

             Returns x, y, y_dom in similar shapes as input
        """

        # Get them to interpretable shapes
        y_a = y_a.reshape(x_a.shape).transpose(1, 0)
        y_b = y_b.reshape(x_b.shape).transpose(1, 0)
        x_a = x_a.transpose(1, 0)
        x_b = x_b.transpose(1, 0)

        b_bs, b_sl = x_a.shape[0], min(x_a.shape[1], x_b.shape[1])

        # Concatenate to make an x and y
        x = np.concatenate((x_a[:, :b_sl], x_b[:, :b_sl]))
        y = np.concatenate((y_a[:, :b_sl], y_b[:, :b_sl]))

        # Shuffle and remember shuffle index to make y labels for domain agnostic training
        intrp = np.arange(b_bs * 2)
        np.random.shuffle(intrp)
        y_dom = (intrp >= b_bs) * 1
        x = x[intrp]
        y = y[intrp]

        x = x.transpose(1, 0)
        y = y.transpose(1, 0).reshape(np.prod(y.shape))

        return x, y, y_dom


'''
    Model definitions
'''


class CustomEncoder(lm_rnn.RNN_Encoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([torch.nn.ModuleList([self.rnns[0], self.dropouths[0]]),
                                    torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
                                    torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])])


class CustomDecoder(text.LinearDecoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([self.decoder, self.dropout])

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class CustomLinear(lm_rnn.PoolingLinearClassifier):

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for i, l in enumerate(self.layers):
            l_x = l(x)
            if i != len(self.layers) -1:
                x = F.relu(l_x)
            else:
                x = torch.sigmoid(l_x)
        return l_x, raw_outputs, outputs


class LanguageModel(nn.Module):

    def __init__(self,
                 _parameter_dict,
                 _device,
                 _encargs,
                 _wgts_e=None,
                 _wgts_d=None):
        super(LanguageModel, self).__init__()

        self.parameter_dict = _parameter_dict
        self.device = _device

        self.encoder = CustomEncoder(**_encargs).to(self.device)
        if _wgts_e:
            self.encoder.load_state_dict(_wgts_e)
        """
            Explanation:
                400*3 because input is [ h_T, maxpool, meanpool ]
                0.4, 0.1 are drops at various layersLM_PATH
        """
        self.linear_dec = CustomDecoder(
            _encargs['ntoken'],
            n_hid=400,
            dropout=params.decoder_drops,
            tie_encoder=self.encoder.encoder,
            bias=False
        ).to(self.device)

        # self.linear_dom = CustomLinear(
        #     2,
        #     n_hid=400,
        #     dropout=0.1 * 0.7,
        #     #             tie_encoder=self.encoder.encoder,
        #     bias=False
        # ).to(self.device)
        self.linear_dom = CustomLinear(layers=params.domclas_layers, drops=params.domclas_drops).to(self.device)
        self.encoder.reset()

    def forward(self, x):
        x_enc = self.encoder(x)
        return self.linear_dec(x_enc)

    def domain(self, x_enc):
        x_enc = list(x_enc)
        x_enc[1] = [GradReverse.apply(enc_tensr) for enc_tensr in x_enc[1]]
        return self.linear_dom(x_enc)[0]

    @property
    def layers(self):
        # layers = [x for x in self.encoder.layers]
        # layers.append(torch.nn.ModuleList([x for x in self.linear_dec.layers]))
        # layers.append(torch.nn.ModuleList([x for x in self.linear_dom.layers]))
        # return torch.nn.ModuleList(layers)
        return self.encoder.layers.extend(self.linear_dec.layers).extend(self.linear_dom.layers)

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            pred = self.forward(x)
            self.train()
            return pred


def _eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())


'''
    Training loop
    
        - sample from wiki
            - e(x_w)        forward from encoder
            - d(e(x_w))     forward from decoder
            - backward normally
            
            - e(x_w)'       grad reverse layer
            - d'(e(x_w)')   forward from domain agnostic layer
            - backward
            
        - sample from imdb
            - same...
'''
def generic_loop(epochs: int,
                 device: torch.device,
                 opt: torch.optim,
                 loss_main_fn: torch.nn,
                 loss_aux_fn: torch.nn,
                 loss_aux_scale: float,
                 model: torch.nn.Module,
                 train_fn: Callable,
                 train_aux_fn: Callable,
                 predict_fn: Callable,
                 data_a: dict,
                 data_b: dict,
                 data_fn: classmethod,
                 save_params,
                 save_dir: Path = None,
                 save_best: bool = False,
                 save_above_trn: float = -np.inf,
                 save_above_aux: float = np.inf,
                 epoch_count: int = 0,
                 epoch_start_hook: Callable = None,
                 epoch_end_hook: Callable = None,
                 batch_start_hook: Callable = None,
                 batch_end_hook: Callable = None,
                 weight_decay: float = params.weight_decay,
                 clip_grads_at: float = params.clip_grads_at,
                 lr_schedule = None,
                 eval_fn: Callable = None,
                 eval_aux_fn: Callable = None,
                 notify: bool = False,
                 notify_key: str = None) -> (list, list, list):
    """

        A generic training loop, which based on diff hook fns (defined below), should handle anything given to it.

        The model need not be an nn.Module,
             but should have correctly wired forward and a predict function.

        # Data input
            Data should be a dict like so:
                {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

            Train_fn must return both loss and y_pred

        # Saving Logic
            There are two conditions on which it saves models (named differently).
            1. Highest train accuracy
            2. Lowest auxiliary accuracy (which is empirically seen to stabilize after two epochs) after two epochs have passed

    :param epochs: number of epochs to train for
    :param data_a: data dict (structure specified above) (main)
    :param data_b: data dict (structure specified above) (aux)
    :param device: torch device to init the tensors with
    :param opt: torch optimizer, with proper param_groups for better lr decay per laye
    :param loss_main_fn: torch.nn loss fn for the actual thing
    :param loss_aux_fn: torch.nn loss fn for the domain agnostic thing
    :param loss_aux_scale: float signifying how much to scale DANN loss with, while combining losses.
    :param model: torch module (for grad clipping)
    :param train_fn: a function which takes x & y, returns loss and y_pred
    :param train_aux_fn: a function which takes x & y, returns loss and y_pred_aux
    :param predict_fn: a fn which takes x and returns y_pred
    :param epoch_count: an int which is added with #epochs (for better representation of how many epochs have actually passed)
            You can use this for when you run the loop say 3 times, do something else and run it for another 10.
    :param epoch_start_hook: a fn that can be called @ start of every epoch (returns model, opt)
    :param epoch_end_hook: a fn that can be called @ end of every epoch (returns model, opt)
    :param batch_start_hook: a fn that can be called @ start of every batch (returns model, opt)
    :param batch_end_hook: a fn that can be called @r end of every batch (returns model, opt)
    :param weight_decay: a L2 ratio (as mentioned in (https://arxiv.org/pdf/1711.05101.pdf)
    :param clip_grads_at: in case you want gradients clipped, send the max val here
    :param lr_schedule: a schedule that is called @ every batch start.
    :param save_best: bool which wants either doesn't save, or saves at best
    :param save_dir: Path object to which we save stuff (based on save_best)
    :param save_params: a dict of all the params used while running and training the model.
    :param save_above_trn: [OPTIONAL] acts as threshold regarading model saving. If the current trn accuracy is less than this, won't.
    :param save_above_aux: [OPTIONAL] acts as threshold regarading model saving. If the current aux accuracy is more than this, won't.
    :param data_fn: a class to which we can pass X and Y, and get an iterator.
    :param eval_fn: function which when given pred and true, returns acc
    :param eval_aux_fn: same as eval_fn but for domain classifier's output.
    :param notify: (optional) flag which enables sending notifications to your phones once the loop is done.
    :param notify_key: (optional) the api key to which the notification is to be sent. You can give it here, or in a file (see README.md)
    :return: traces
    """

    train_loss_main = []
    train_loss_aux = []
    train_acc_main = []
    train_acc_aux = []
    val_acc = []
    lrs = []
    saved_info = {}

    assert (not save_best) or (save_best and save_dir), "No save dir specified."

    # Epoch level
    for e in range(epoch_count, epochs+epoch_count):

        per_epoch_loss_main = []
        per_epoch_loss_aux = []
        per_epoch_tr_acc_main = []
        per_epoch_tr_acc_aux = []

        # Train
        with Timer() as timer:

            model.train()

            # @TODO: Add hook at start of epoch (how to decide what goes in)
            if epoch_start_hook: epoch_start_hook()

            # Make data
            trn_dl = data_fn(data_a=data_a['train'], data_b=data_b['train'])
            val_dl = data_fn(data_a=data_a['valid'], data_b=data_b['valid'])

            for x, y, y_aux in tqdm(trn_dl):

                if batch_start_hook: batch_start_hook()
                opt.zero_grad()

                if lr_schedule: lrs.append(update_lr(opt, lr_schedule.get()))

                # 0. Convert np arrs to torch tensors
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)
                _y_aux = torch.tensor(y_aux, dtype=torch.long, device=device)

                # A: Normal stuff
                op = train_fn(_x)
                y_pred = op[0]
                loss_main = loss_main_fn(y_pred, _y)

                # B: Domain agnostic stuff
                y_pred_aux = train_aux_fn(op[1:])
                loss_aux = loss_aux_fn(y_pred_aux, _y_aux)

                # C. Add losses with scale.
                loss = loss_main + (loss_aux_scale * loss_aux)

                # Logging
                per_epoch_tr_acc_main.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_tr_acc_aux.append(eval_aux_fn(y_pred=y_pred_aux, y_true=_y_aux).item())
                per_epoch_loss_main.append(loss_main.item())
                per_epoch_loss_aux.append(loss_aux.item())

                # Pass aux gradients
                loss.backward(retain_graph=False)

                # Optimizer Step
                if clip_grads_at > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads_at)
                for group in opt.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-weight_decay * group['lr'], param.data)

                opt.step()
                if batch_end_hook: batch_end_hook()

            if epoch_end_hook: epoch_end_hook()

        # Val
        with torch.no_grad():

            model.eval()

            per_epoch_vl_acc = []
            for x, y, y_aux in tqdm(val_dl):
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                y_pred = predict_fn(_x)[0]

                per_epoch_vl_acc.append(eval_fn(y_pred, _y).item())

        # Bookkeep
        train_acc_main.append(np.mean(per_epoch_tr_acc_main))
        train_acc_aux.append(np.mean(per_epoch_tr_acc_aux))
        train_loss_main.append(np.mean(per_epoch_loss_main))
        train_loss_aux.append(np.mean(per_epoch_loss_main))
        val_acc.append(np.mean(per_epoch_vl_acc))

        print("Epoch: %(epo)03d | "
              "Loss: %(loss).4f | "
              "Loss_aux: %(loss_aux).4f | "
              "Tr_c: %(tracc)0.4f | "
              "Vl_c: %(vlacc)0.5f | "
              "Tr_aux: %(tracc_aux)0.4f | "
              " Time: %(time).3f m"
              % {'epo': e+epoch_count,
                 'loss': float(np.mean(per_epoch_loss_main)),
                 'loss_aux': float(np.mean(per_epoch_loss_aux)),
                 'tracc': float(np.mean(per_epoch_tr_acc_main)),
                 'tracc_aux': float(np.mean(per_epoch_tr_acc_aux)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

        # Save block (flag and condition)
        if save_best and train_acc_main[-1] >= save_above_trn:

            # Update threshold
            save_above_trn = train_acc_main[-1]

            # Adding epoch info along with options
            save_params.epoch = e

            # Call save function and save
            mt_save(save_dir,
                    torch_stuff=[tosave('unsup_model_hightrn.torch', model.state_dict()),
                                 tosave('unsup_model_enc_hightrn.torch', model.encoder.state_dict())],
                    pickle_stuff=[tosave('unsup_traces.pkl', [train_acc_main, val_acc, train_acc_aux, train_loss_main, train_loss_aux, lrs]),
                                  tosave('unsup_options.pkl', save_params)])
            print(f"Model saved on Epoch {e} at {save_dir} because of highest training acc so far")

            # Log the saved thing
            saved_info['epoch'] = e
            saved_info['accuracy'] = val_acc[-1]
            saved_info['directory'] = save_dir

        # Save block (flag and condition) (When more than 2 epochs have pased and we see lowest aux accuracy)
        if save_best and e > 1 and train_acc_aux[-1] <= save_above_aux:

            # Update threshold
            save_above_aux = train_acc_aux[-1]

            # Adding epoch info along with options
            save_params.epoch = e

            # Call save function and save
            mt_save(save_dir,
                    torch_stuff=[tosave('unsup_model_lowaux.torch', model.state_dict()),
                                 tosave('unsup_model_enc_lowaux.torch', model.encoder.state_dict())],
                    pickle_stuff=[
                        tosave('unsup_traces.pkl', [train_acc_main, val_acc, train_acc_aux, train_loss_main, train_loss_aux, lrs]),
                        tosave('unsup_options.pkl', save_params)])
            print(f"Model saved on Epoch {e} at {save_dir} because of lowest auxiliary accuracy so far")

            # Log the saved thing
            saved_info['epoch'] = e
            saved_info['accuracy'] = val_acc[-1]
            saved_info['directory'] = save_dir

    if notify:
        if not saved_info:
            message_template = "Your model is done training."
            send_notification(data=saved_info, key=notify_key, message_template=message_template)
        else:
            send_notification(data=saved_info, key=notify_key)

    return train_acc_main, val_acc, train_acc_aux, train_loss_main, train_loss_aux, lrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Domain adversarial for ULMFiT\'s language models')
    parser.add_argument("-t", "--quick", type=bool, required=False,
                        help="True if you want to only train on first 1000 train,test samples")
    parser.add_argument("-d", "--debug", type=bool, required=False, help="True if you want a verbose run")
    parser.add_argument("-sf", "--safemode", type=bool, required=False, help="True if you dont want to save anything")
    parser.add_argument("-m", "--message", type=str, required=False, help="Message to be saved alongwith traces", default=None)
    parser.add_argument("-p", "--pretrained", type=bool, required=False,
                        help="False if you don't want to load pretrained weights in LM")

    parse_args = vars(parser.parse_args())
    QUICK, DEBUG, PRETRAINED = parse_args['quick'], parse_args['debug'], parse_args['pretrained']
    MESSAGE = parse_args['message']
    SAFE_MODE =  parse_args['safemode']

    params.message = MESSAGE
    params.quick = QUICK

    if DEBUG:
        print("Pulling data from disk")

    # Pulling data from disk
    data_puller = DataPuller(debug=False, max_vocab=params.max_vocab_task, min_freq=params.min_vocab_freq, trim_trn=1000, trim_val=-1)
    trn_lm, val_lm, _ = data_puller.get('imdb', supervised=False, trim=params.quick)
    wiki_trn_lm, wiki_val_lm, itos = data_puller.get('wikitext', supervised=False, trim=params.quick, merge_vocab=params.max_vocab_wiki)
    vs = len(itos)

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

    # noinspection PyCallingNonCallable
    wgts['0.encoder.weight'] = T(new_w)
    # noinspection PyCallingNonCallable
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    # noinspection PyCallingNonCallable
    wgts['1.decoder.weight'] = T(np.copy(new_w))
    wgts_enc = {'.'.join(k.split('.')[1:]): val
                for k, val in wgts.items() if k[0] == '0'}
    wgts_dec = {'.'.join(k.split('.')[1:]): val
                for k, val in wgts.items() if k[0] == '1'}

    '''
        Setting up things for training.
    '''
    bptt = 70
    bs = params.bs
    opt_fn = partial(torch.optim.SGD)  # , betas=params.adam_betas)  # @TODO: find real optimizer, and params

    # Load the pre-trained model
    parameter_dict = {'itos2': itos2}
    dps = params.encoder_drops
    encargs = {'ntoken': new_w.shape[0],
               'emb_sz': 400, 'n_hid': 1150,
               'n_layers': 3, 'pad_token': 0,
               'qrnn': False, 'dropouti': dps[0],
               'wdrop': dps[2], 'dropoute': dps[3], 'dropouth': dps[4]}

    lm = LanguageModel(parameter_dict, device, _wgts_e=wgts_enc if PRETRAINED else None, _wgts_d=wgts_dec, _encargs=encargs)
    opt = make_opt(lm, opt_fn, lr=params.lr.init)
    loss_main_fn = F.cross_entropy
    loss_aux_fn = F.cross_entropy

    # Make data
    data_fn_unidomain = partial(text.LanguageModelLoader, bs=bs, bptt=bptt)
    data_imdb = {'train': np.concatenate(trn_lm), 'valid': np.concatenate(val_lm)}
    data_wiki = {'train': np.concatenate(wiki_trn_lm), 'valid': np.concatenate(wiki_val_lm)}
    data_fn = partial(DomainAgnosticSampler, data_fn=data_fn_unidomain)

    # Set up lr and freeze stuff
    for grp in opt.param_groups:
        grp['lr'] = 0.0
    opt.param_groups[3]['lr'] = params.lr.init
    opt.param_groups[4]['lr'] = params.lr.init

    # lr_args = {'batches':, 'cycles': 1}
    lr_args = {'iterations': len(data_fn(data_a=data_wiki['train'], data_b=data_imdb['train'])),
               'cut_frac': params.lr.sltr_cutfrac, 'ratio': params.lr.sltr_ratio}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.SlantedTriangularLR)

    # Find places to save model
    save_dir = mt_save_dir(PATH / 'models', _newdir=True) if not SAFE_MODE else ''

    if not SAFE_MODE:
        # Start to put permanent things there, like the itos
        mt_save(save_dir,
                pickle_stuff=[tosave('itos.pkl', itos)])

    args = {'epochs': 1, 'weight_decay': params.weight_decay, 'data_a': data_imdb, 'data_b': data_wiki,
            'device': device, 'opt': opt, 'loss_main_fn': loss_main_fn, 'loss_aux_fn': loss_aux_fn,
            'train_fn': lm, 'train_aux_fn': lm.domain, 'predict_fn': lm.predict, 'data_fn': data_fn, 'model': lm,
            'eval_fn': _eval, 'eval_aux_fn': _eval, 'batch_start_hook': partial(mtlp.reset_hidden, lm),
            'clip_grads_at': params.clip_grads_at, 'lr_schedule': lr_schedule, 'loss_aux_scale': params.loss_scale,
            'save_dir': save_dir, 'save_best': not SAFE_MODE, 'save_params': params}

    '''
        Actual training
    '''
    # print("Time taken to get everything so far done")
    traces_start = generic_loop(**args)

    # Now unfreeze all layers and apply discr
    for grp in opt.param_groups:
        grp['lr'] = params.lr.init

    lr_dscr = lambda opt, lr, fctr=params.lr.dscr: [lr / (fctr ** i) for i in range(len(opt.param_groups))[::-1]]
    update_lr(opt, lr_dscr(opt, params.lr.init))

    if DEBUG:
        print([x['lr'] for x in opt.param_groups])

    lr_args = {'iterations': len(data_fn(data_a=data_wiki['train'], data_b=data_imdb['train'])) * 15,
               'cut_frac': params.lr.sltr_cutfrac, 'ratio': params.lr.sltr_ratio}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.SlantedTriangularLR)
    args['save_above_trn'] = np.max(traces_start[0])
    # args['save_above_aux'] = np.min(traces_start[2][2:])  # Not updating this var since we ignore the DANN acc of the first few epochs anyway
    args['lr_schedule'] = lr_schedule
    args['epochs'] = 15
    args['epoch_count'] = 1
    args['notify'] = True

    traces_main = generic_loop(**args)
    traces = [a + b for a, b in zip(traces_start, traces_main)]

    # Final save, just in case
    # Dumping stuff
    if not SAFE_MODE:
        mt_save(save_dir, message=MESSAGE,
                torch_stuff=[tosave('unsup_model_final.torch', lm.state_dict()),
                             tosave('unsup_model_enc_final.torch', lm.encoder.state_dict())],
                pickle_stuff=[tosave('final_unsup_traces.pkl', traces), tosave('unsup_options.pkl', params)])

    # Interpreting Traces
    trn_best = np.max(traces[0])
    trn_best_ = np.argmax(traces[0])
    val_attrn = traces[1][trn_best_]
    val_best = np.max(traces[1])
    val_best_ = np.argmax(traces[1])
    aux_attrn = traces[2][trn_best_]
    aux_best = np.min(traces[2][2:])
    aux_best_ = np.argmin(traces[2][2:])
    print(f"Train Best: {trn_best:.4f} at {trn_best_}\n"
          f"Valid @Trn: {val_attrn:.4f}\n"
          f"Valid Best: {val_best:.4f} at {val_best_}\n"
          f"DomAg @Trn: {aux_attrn:.4f}\n"
          f"DomAg Best: {aux_best:.4f} at {aux_best_}")

