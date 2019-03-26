# noinspection PyShadowingNames
# External Lib imports
import collections
from functools import partial
from typing import Tuple, Union, Callable

# Torch imports
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as func

# Mytorch imports
from mytorch import loops as mtlp
from mytorch.utils.goodies import *
from mytorch import lriters as mtlr

# Local imports
from utils import dann_loop
from data import DataPuller
from options import Phase2 as params

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, core, lm_rnn


DEVICE = 'cuda:2'
KNOWN_DATASETS = ['imdb', 'wikitext']

device = torch.device(DEVICE)
np.random.seed(42)
torch.manual_seed(42)

# Path fields
PATH = Path('resources/proc/imdb')
DATA_PROC_PATH = PATH / 'data'
DATA_LM_PATH = PATH / 'datalm'
DUMP_PATH = Path('resources/models/runs')

LM_PATH = Path('resources/models')
LM_PATH.mkdir(exist_ok=True)
PRE_PATH = LM_PATH / 'wt103'
PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'

'''
    Data sampler for this training
'''


# noinspection PyShadowingNames
class DomainAgnosticSampler:
    """ Sample data for language model training from two different domains in one batch. """

    def __init__(self, data_fn: Callable, data: Tuple[Union[list, np.ndarray], Union[list, np.ndarray]]):
        """
            Here, data_fn would be something like
                `partial(text.LanguageModelLoader, bs=bs, bptt=bptt)`
            And data_a/b would be something like
                `{'train': np.concatenate(trn_lm), 'valid': np.concatenate(val_lm)}['train']`
            data_fn (fastai's language model loader) flattens y and returns x of seqlen, batchsize
        """
        data_a, data_b = data
        self.args = {'data_fn': data_fn, 'data_a': data_a, 'data_b': data_b}
        self.reset(**self.args)
        self.itera, self.iterb = iter([]), iter([])

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

    def forward(self, input, domain=None):
        """ Overwrote fn to keep the interface same b/w phase 2 & phase 3 models (same training loop)"""
        super().forward(input)

    @property
    def layers(self):
        return torch.nn.ModuleList([torch.nn.ModuleList([self.rnns[0], self.dropouths[0]]),
                                    torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
                                    torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])])


class CustomDecoder(text.LinearDecoder):

    @property
    def layers(self):
        return torch.nn.ModuleList([self.decoder, self.dropout])

    def forward(self, x):
        raw_outputs, outputs = x
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


# noinspection PyShadowingNames
class CustomLinear(lm_rnn.PoolingLinearClassifier):

    def forward(self, x):
        raw_outputs, outputs = x
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for i, l in enumerate(self.layers):
            l_x = l(x)
            if i != len(self.layers) -1:
                x = func.relu(l_x)
            else:
                x = torch.sigmoid(l_x)
        # noinspection PyUnboundLocalVariable
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Domain adversarial for ULMFiT\'s language models')
    parser.add_argument("-t", "--quick", type=str2bool, required=False,
                        help="True if you want to only train on first 1000 train,test samples")
    parser.add_argument("--debug", type=str2bool, required=False,
                        help="True if you want a verbose run")
    parser.add_argument("-sf", "--safemode", type=str2bool, required=False,
                        help="True if you dont want to save anything")
    parser.add_argument("-m", "--message", type=str, required=False,
                        help="Message to be saved alongwith traces", default=None)
    parser.add_argument("-p", "--pretrained", type=str2bool, required=False,
                        help="False if you don't want to load pretrained weights in LM")
    parser.add_argument("-d", "--datasets", type=str, required=False, default="imdb,wikitext",
                        help="Comma separated two dataset names like wikitext,imdb")

    parse_args = vars(parser.parse_args())
    QUICK = parse_args['quick']
    DEBUG = parse_args['debug']
    PRETRAINED = parse_args['pretrained']
    MESSAGE = parse_args['message']
    SAFE_MODE = parse_args['safemode']
    DATASETS = parse_args['datasets'].split(',')

    # Check args.
    for dataset in DATASETS:
        assert dataset in KNOWN_DATASETS, f"Couldn't find a dataset called {dataset}. Exiting."

    params.message = MESSAGE
    params.quick = QUICK

    if DEBUG:
        print("Pulling data from disk")

    # Pulling data from disk
    data_puller = DataPuller(debug=False, max_vocab=params.max_vocab_task, min_freq=params.min_vocab_freq, trim_trn=1000, trim_val=-1)
    trn_lm, val_lm, _ = data_puller.get('imdb', supervised=False, trim=params.quick, cached=not params.quick)
    wiki_trn_lm, wiki_val_lm, itos = data_puller.get('wikitext', supervised=False, trim=params.quick,
                                                     merge_vocab=params.max_vocab_wiki, cached=not params.quick)
    vs = len(itos)

    """
        Now we pull pretrained models from disk    
    """

    if DEBUG:
        print("Pulling models from disk")

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
    wgts['0.encoder.weight'] = tensor(new_w)
    # noinspection PyCallingNonCallable
    wgts['0.encoder_with_dropout.embed.weight'] = tensor(np.copy(new_w))
    # noinspection PyCallingNonCallable
    wgts['1.decoder.weight'] = tensor(np.copy(new_w))
    wgts_enc = {'.'.join(k.split('.')[1:]): val
                for k, val in wgts.items() if k[0] == '0'}
    wgts_dec = {'.'.join(k.split('.')[1:]): val
                for k, val in wgts.items() if k[0] == '1'}

    '''
        Setting up things for training.
    '''
    bptt = 70
    bs = params.bs
    opt_fn = partial(torch.optim.SGD)  # , betas=params.adam_betas)

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
    loss_main_fn = func.cross_entropy
    loss_aux_fn = func.cross_entropy

    # Make data
    data_fn_unidomain = partial(text.LanguageModelLoader, bs=bs, bptt=bptt)
    data_train = (np.concatenate(trn_lm), np.concatenate(wiki_trn_lm))
    data_valid = (np.concatenate(val_lm), np.concatenate(wiki_val_lm))
    data = {'train': data_train, 'valid': data_valid}
    data_fn = partial(DomainAgnosticSampler, data_fn=data_fn_unidomain)

    # Set up lr and freeze stuff
    for grp in opt.param_groups:
        grp['lr'] = 0.0
    opt.param_groups[3]['lr'] = params.lr.init
    opt.param_groups[4]['lr'] = params.lr.init

    # lr_args = {'batches':, 'cycles': 1}
    lr_args = {'iterations': len(data_fn(data=data['train'])),
               'cut_frac': params.lr.sltr_cutfrac, 'ratio': params.lr.sltr_ratio}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.SlantedTriangularLR)

    # Find places to save model
    save_dir = mt_save_dir(DUMP_PATH / 'models', _newdir=True) if not SAFE_MODE else ''
    save_fnames = {'torch_stuff':
                       {'hightrn':
                            {'model': 'unsup_model_hightrn.torch',
                             'enc': 'unsup_model_hightrn_enc.torch'},
                        'lowaux':
                            {'model': 'unsup_model_lowaux.torch',
                             'enc': 'unsup_model_lowaux_enc.torch'}}}

    if not SAFE_MODE:
        # Start to put permanent things there, like the itos
        mt_save(save_dir,
                pickle_stuff=[tosave('itos.pkl', itos)])

    args = {'epochs': 1, 'weight_decay': params.weight_decay, 'data': data,
            'device': device, 'opt': opt, 'loss_main_fn': loss_main_fn, 'loss_aux_fn': loss_aux_fn,
            'train_fn': lm, 'train_aux_fn': lm.domain, 'predict_fn': lm.predict, 'data_fn': data_fn, 'model': lm,
            'eval_fn': _eval, 'eval_aux_fn': _eval, 'batch_start_hook': partial(mtlp.reset_hidden, lm),
            'clip_grads_at': params.clip_grads_at, 'lr_schedule': lr_schedule, 'loss_aux_scale': params.loss_scale,
            'save_dir': save_dir, 'save_best': not SAFE_MODE, 'save_params': params, 'save_fnames': save_fnames}

    '''
        Actual training
    '''
    # print("Time taken to get everything so far done")
    traces_start = dann_loop(**args)

    # Now unfreeze all layers and apply discr
    for grp in opt.param_groups:
        grp['lr'] = params.lr.init

    lr_dscr = lambda optim, lr, fctr=params.lr.dscr: [lr / (fctr ** p_grp) for p_grp in range(len(optim.param_groups))[::-1]]
    update_lr(opt, lr_dscr(opt, params.lr.init))

    if DEBUG:
        print([x['lr'] for x in opt.param_groups])

    lr_args = {'iterations': len(data_fn(data=data['train'])) * 15,
               'cut_frac': params.lr.sltr_cutfrac, 'ratio': params.lr.sltr_ratio}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.SlantedTriangularLR)
    args['save_above_trn'] = np.max(traces_start[0])
    # args['save_above_aux'] = np.min(traces_start[2][2:])  # Not updating this var since we ignore the DANN acc of the first few epochs anyway
    args['lr_schedule'] = lr_schedule
    args['epochs'] = 15
    args['epoch_count'] = 1
    args['notify'] = True

    traces_main = dann_loop(**args)
    traces = [a + b for a, b in zip(traces_start, traces_main)]

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

