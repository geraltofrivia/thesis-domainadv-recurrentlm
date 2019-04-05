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


DEVICE = 'cuda'
KNOWN_DATASETS = ['imdb', 'wikitext', '']

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

    def __init__(self, data: Tuple[Union[list, np.ndarray], Union[list, np.ndarray]], data_fn: Callable):
        """
            Here, data_fn would be something like
                `partial(text.LanguageModelLoader, bs=bs, bptt=bptt)`
            And data_a/b would be something like
                `{'train': np.concatenate(trn_lm), 'valid': np.concatenate(val_lm)}['train']`
            data_fn (fastai's language model loader) flattens y and returns x of seqlen, batchsize
        """
        self.args = {'data_fn': data_fn, 'data': data,}
        self.iters = [iter([]) for _ in range(len(data))]
        self.reset(**self.args)

    def reset(self, data_fn: Callable, data: list):
        self.iters = [iter(data_fn(data_)) for data_ in data]

    def __iter__(self):
        return self

    def __next__(self):
        x,y = [], []
        for iter_ in self.iters:
            x_, y_ = iter_.__next__()
            x.append(x_)
            y.append(y_)
        return self._combine_batch_(x, y)

    def __len__(self):
        return min([len(self.args['data_fn'](data_)) for data_ in self.args['data'] ])

    @staticmethod
    def _combine_batch_(x, y):
        """
            :param x is a list of np.arr looks like seqlen, batchsize
            :param y is a corresponding list of np.arr (one word ahead than x_a) which is a flattened x_a.shape mat

             Returns x, y, y_dom in similar shapes as input
        """

        # Get them to interpretable shapes
        y = [y_.reshape(x[i].shape).transpose(1, 0) for i,y_ in enumerate(y)]
        x = [x_.transpose(1, 0) for x_ in x]

        b_bs, b_sl = x[0].shape[0], min([x_.shape[1] for x_ in x])

        # Concatenate to make an x and y
        x = np.concatenate([x_[:, :b_sl] for x_ in x])
        y = np.concatenate([y_[:, :b_sl] for y_ in y])

        # Shuffle and remember shuffle index to make y labels for domain agnostic training
        intrp = np.arange(b_bs * 2)
        np.random.shuffle(intrp)
        y_dom = (intrp >= b_bs) * 1
        x = x[intrp]
        y = y[intrp]

        x = x.transpose(1, 0)
        y = y.transpose(1, 0).reshape(np.prod(y.shape))

        return x, y, y_dom


class CustomLanguageModelLoader(text.LanguageModelLoader):
    """ Overwriting the class so we can call it within the same way of iterating over data as in other cases."""

    def __init__(self, data, **args):
        super().__init__(data, **args)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            _res = list(res) + [torch.zeros(res[0].shape[1])]
            self.i += seq_len
            self.iter += 1
            yield res


'''
    Model definitions
'''


class CustomEncoder(lm_rnn.RNN_Encoder):

    def forward(self, input, domain=None):
        """ Overwrote fn to keep the interface same b/w phase 2 & phase 3 models (same training loop)"""
        return super().forward(input)

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
        return result, (raw_outputs, outputs)


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
            if i != len(self.layers) - 1:
                x = func.relu(l_x)
            else:
                x = torch.sigmoid(l_x)
        # noinspection PyUnboundLocalVariable
        return l_x, (raw_outputs, outputs)


class LanguageModel(nn.Module):

    def __init__(self,
                 _parameter_dict,
                 _device,
                 _encargs,
                 _n_tasks=2,
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

        self.linear_dom = CustomLinear(layers=params.domclas_layers + [_n_tasks], drops=params.domclas_drops).to(self.device)
        self.encoder.reset()

    def forward(self, x, d):
        """ d is not used (only so the loop remains same b/w phase 2 and phase 3 models) """
        x_enc = self.encoder(x, d)
        return self.linear_dec(x_enc)

    def domain(self, x_enc):
        x_enc = list(x_enc)
        x_enc[1] = [GradReverse.apply(enc_tensr) for enc_tensr in x_enc[1]]
        return self.linear_dom(x_enc)[0]

    @property
    def layers(self):
        return self.encoder.layers.extend(self.linear_dec.layers).extend(self.linear_dom.layers)

    def predict(self, x, d):
        with torch.no_grad():
            self.eval()
            pred = self.forward(x, d)
            self.train()
            return pred


def _eval(y_pred, y_true, tasks: int=1, task_index: torch.tensor=None):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())


def loss_wrapper(y_pred, y_true, loss_fn, **args):
    return loss_fn(y_pred, y_true)


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


    for dataset in DATASETS:
        assert dataset in KNOWN_DATASETS, f"Couldn't find a dataset called {dataset}. Exiting."

    params.message = MESSAGE
    params.quick = QUICK
    params.datasets = DATASETS
    if len(DATASETS) < 2: params.loss_scale = 0.0

    if DEBUG:
        print("Pulling data from disk")

    # Pulling data from disk
    data_puller = DataPuller(debug=False, max_vocab=params.max_vocab_task, min_freq=params.min_vocab_freq, trim_trn=1000, trim_val=-1)

    trn_lm, val_lm = [], []
    for dataset in DATASETS:

        trn_lm_, val_lm_, itos = data_puller.get(dataset, supervised=False, trim=params.quick, cached=True, merge_vocab=params.max_vocab_others)

        # Append data to main lists
        trn_lm.append(trn_lm_)
        val_lm.append(val_lm_)

    vs = len(itos)

    """
        Now we pull pretrained models from disk    
    """

    if DEBUG:
        print("Pulling models from disk")

    em_sz, nh, nl = 400, 1150, 3
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
    lengths = np.array([len(CustomLanguageModelLoader(np.concatenate(trn_lm_), bs=bs, bptt=bptt)) for trn_lm_ in trn_lm])
    # l_a, l_b = len(text.LanguageModelLoader(np.concatenate(trn_lm), bs=bs, bptt=bptt)), \
    #            len(text.LanguageModelLoader(np.concatenate(wiki_trn_lm), bs=bs, bptt=bptt))
    weights = torch.tensor(np.ascontiguousarray((lengths/np.sum(lengths))[::-1]), dtype=torch.float, device=device) if len(DATASETS) > 1 else None

    # Load the pre-trained model
    parameter_dict = {'itos2': itos2}
    dps = params.encoder_drops
    encargs = {'ntoken': new_w.shape[0],
               'emb_sz': 400, 'n_hid': 1150,
               'n_layers': 3, 'pad_token': 0,
               'qrnn': False, 'dropouti': dps[0],
               'wdrop': dps[2], 'dropoute': dps[3], 'dropouth': dps[4]}

    lm = LanguageModel(parameter_dict, device, _encargs=encargs, _n_tasks=len(DATASETS),
                       _wgts_e=wgts_enc if PRETRAINED else None, _wgts_d=wgts_dec)
    opt = make_opt(lm, opt_fn, lr=params.lr.init)
    loss_main_fn = partial(loss_wrapper, loss_fn=func.cross_entropy)
    loss_aux_fn = partial(loss_wrapper, loss_fn=nn.CrossEntropyLoss(weights))

    # Make data
    if len(DATASETS) > 1:
        data_fn_unidomain = partial(text.LanguageModelLoader, bs=bs, bptt=bptt)
        data_train = [np.concatenate(trn_lm_) for trn_lm_ in trn_lm]
        data_valid = [np.concatenate(val_lm_) for val_lm_ in val_lm]
        data = {'train': data_train, 'valid': data_valid}
        data_fn = partial(DomainAgnosticSampler, data_fn=data_fn_unidomain)
    else:
        data_fn_unidomain = partial(CustomLanguageModelLoader, bs=bs, bptt=bptt)
        data_train = [np.concatenate(trn_lm_) for trn_lm_ in trn_lm][0]
        data_valid = [np.concatenate(val_lm_) for val_lm_ in val_lm][0]
        data = {'train': data_train, 'valid': data_valid}
        data_fn = data_fn_unidomain

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
    save_dir = mt_save_dir(DUMP_PATH / '_'.join(DATASETS), _newdir=True) if not SAFE_MODE else ''
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
            'eval_fn': _eval, 'eval_aux_fn': _eval, 'batch_start_hook': partial(mtlp.reset_hidden, lm), 'tasks': 2,
            'clip_grads_at': params.clip_grads_at, 'lr_schedule': lr_schedule, 'loss_aux_scale': params.loss_scale,
            'save_dir': save_dir, 'save': not SAFE_MODE, 'save_params': params, 'save_fnames': save_fnames}

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

