"""
    Pulls an unsupervised fine tuned model from disk, also data, and goes to town on it.

    @TODO: Add embeddings in layer
    @TODO: Check if LR is reset after its fucked up by sltr
"""

# External Lib imports
import os
import pandas as pd
from functools import partial

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, lm_rnn

# Torch imports
import torch.nn as nn
import torch.optim as optim

# Mytorch imports
from mytorch import loops, lriters as mtlr, dataiters as mtdi
from mytorch.utils.goodies import *

# Local imports
from data import DataPuller
from options import Phase3 as params

device = torch.device('cuda')
np.random.seed(42)
torch.manual_seed(42)

'''
    Paths and macros
'''
PATH = Path('resources/proc/imdb')
DATA_PROC_PATH = PATH / 'data'
DATA_LM_PATH = PATH / 'datalm'

LM_PATH = Path('resources/models')
LM_PATH.mkdir(exist_ok=True)
PRE_PATH = LM_PATH / 'wt103'
PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'


'''
    Model code
'''


class CustomEncoder(lm_rnn.MultiBatchRNN):
    @property
    def layers(self):
        # TODO: ADD ENCODERR!!!!!!!!!!
        return torch.nn.ModuleList([torch.nn.ModuleList([self.rnns[0], self.dropouths[0]]),
                                    torch.nn.ModuleList([self.rnns[1], self.dropouths[1]]),
                                    torch.nn.ModuleList([self.rnns[2], self.dropouths[2]])])


class TextClassifier(nn.Module):

    # @TODO: inject comments.
    def __init__(self,
                 _device: torch.device,
                 ntoken: int,
                 dps: list,
                 enc_wgts = None,
                 _debug=False):
        super(TextClassifier, self).__init__()

        self.device = _device

        # Load the pre-trained model
        args = {'ntoken': ntoken, 'emb_sz': 400, 'n_hid': 1150,
                'n_layers': 3, 'pad_token': 0, 'qrnn': False, 'bptt': 70, 'max_seq': 1400,
                'dropouti': dps[0], 'wdrop': dps[1], 'dropoute': dps[2], 'dropouth': dps[3]}
        self.encoder = CustomEncoder(**args).to(self.device)
        if enc_wgts:
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

    def forward(self, x):
        # inputs are S*B

        # Encoding all the data
        op_p = self.encoder(x.transpose(1, 0))
        # pos_batch = op_p[1][-1][-1]
        score = self.linear(op_p)[0]

        return score

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            predicted = self.forward(x)
            self.train()
            return predicted


def epoch_end_hook() -> None:
    lr_schedule.reset()


# noinspection PyUnresolvedReferences
def _eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())


if __name__ == "__main__":

    # Get args from console
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quick", type=bool, required=False,
                    help="True if you want to only train on first 1000 train,test samples")
    ap.add_argument("-d", "--debug", type=bool, required=False,
                    help="True if you want a verbose run")
    ap.add_argument("-p", "--pretrained", type=bool, required=False,
                    help="True if you want a verbose run")
    ap.add_argument("-sf", "--safemode", type=bool, required=False,
                    help="True if you dont want to save anything")
    ap.add_argument("-ms", "--modelsuffix", default='_lowaux', type=str,
                    help="Input either `_lowaux`;`_hightrn` or nothing depending on which kind of model you want to load.")
    ap.add_argument("-md", "--modeldir", required=True,
                    help="Need to provide the folder name (not the entire dir) to the desired phase 2 model. E.g. `--modeldir 2` shall suffice.")

    args = vars(ap.parse_args())
    QUICK = args['quick']
    DEBUG = args['debug']
    MODEL_NUM = args['modeldir']
    PRETRAINED = args['pretrained']
    MODEL_SUFFIX = args['modelsuffix']
    SAFE_MODE = args['safemode']
    UNSUP_MODEL_DIR = PATH / 'models' / MODEL_NUM

    assert MODEL_SUFFIX in ['_lowaux', '_hightrn', '', '_final'], 'Incorrect Suffix given with which to load model'

    params.quick = QUICK
    params.model_dir = str(UNSUP_MODEL_DIR) + ' and ' + str(MODEL_NUM)
    params.model_suffix = MODEL_SUFFIX

    data_puller = DataPuller(debug=False, max_vocab=params.max_vocab_task, min_freq=params.min_vocab_freq, trim_trn=1000, trim_val=-1)
    trn_texts, trn_labels, val_texts, val_labels, itos = data_puller.get('imdb', supervised=True, trim=params.quick)

    # Lose label 2 from train
    trn_texts = trn_texts[trn_labels < 2]
    trn_labels = trn_labels[trn_labels < 2]

    # Create representations of text using old itos
    itos_path = UNSUP_MODEL_DIR / 'itos.pkl'
    itos2 = pickle.load(itos_path.open('rb'))
    stoi2 = {v: k for k, v in enumerate(itos2)}

    trn_texts = np.array([[stoi2.get(w, 0) for w in para] for para in trn_texts])
    val_texts = np.array([[stoi2.get(w, 0) for w in para] for para in val_texts])

    '''
        Make model
    '''
    dps = list(params.encoder_dropouts)
    # enc_wgts = torch.load(LM_PATH, map_location=lambda storage, loc: storage)
    enc_wgts = torch.load(UNSUP_MODEL_DIR / ('unsup_model_enc' + MODEL_SUFFIX + '.torch'), map_location=lambda storage, loc: storage)
    clf = TextClassifier(device, len(itos2), dps, enc_wgts=enc_wgts if PRETRAINED else None)

    '''
        Setup things for training (data, loss, opt, lr schedule etc
    '''
    bs = params.bs
    loss_fn = torch.nn.CrossEntropyLoss()
    opt_fn = partial(optim.Adam, betas=params.adam_betas)
    opt = make_opt(clf, opt_fn, lr=0.0)
    opt.param_groups[-1]['lr'] = 0.01

    # Make data
    data_fn = partial(mtdi.SortishSampler, _batchsize=bs, _padidx=1)
    data = {'train': {'x': trn_texts, 'y': trn_labels}, 'valid': {'x': val_texts, 'y': val_labels}}

    # Make lr scheduler
    lr_args = {'iterations': len(data_fn(data['train'])), 'cycles': 1}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)

    save_args = {'torch_stuff': [tosave('model.torch', clf.state_dict()), tosave('model_enc.torch', clf.encoder.state_dict())]}

    args = {'epochs': 1, 'epoch_count':0, 'data': data, 'device': device,
            'opt': opt, 'loss_fn': loss_fn, 'model': clf,
            'train_fn': clf, 'predict_fn': clf.predict,
            'epoch_end_hook': epoch_end_hook, 'weight_decay': params.weight_decay,
            'clip_grads_at': params.clip_grads_at, 'lr_schedule': lr_schedule,
            'data_fn': data_fn, 'eval_fn': _eval,
            'save': not SAFE_MODE, 'save_params': params, 'save_dir': UNSUP_MODEL_DIR, 'save_args': save_args}

    '''
        Training schedule:
        
        1. Unfreeze one layer. Train for 1 epoch
        2 - 5. Unfreeze one layer, train for 1 epoch
        3. Train for 15 epochs (after all layers are unfrozen). Use 15 cycles for cosine annealing.
    '''
    # opt.param_groups[-1]['lr'] = 0.01
    traces = loops.generic_loop(**args)

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = loops.generic_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = loops.generic_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    opt.param_groups[-4]['lr'] = 0.001
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = loops.generic_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    opt.param_groups[-4]['lr'] = 0.001
    opt.param_groups[-5]['lr'] = 0.001
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = loops.generic_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    opt.param_groups[-4]['lr'] = 0.001
    opt.param_groups[-5]['lr'] = 0.001
    lr_args['cycles'] = 15
    args['epochs'] = 15
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    args['notify'] = True

    traces_new = loops.generic_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    if not SAFE_MODE:
        mt_save(UNSUP_MODEL_DIR,
                torch_stuff=[tosave('sup_model_final.torch', clf.state_dict())],
                pickle_stuff=[tosave('final_sup_traces.pkl', traces), tosave('unsup_options.pkl', params)])
