"""
    Pulls an unsupervised fine tuned model from disk, also data, and goes to town on it.

    @TODO: Add embeddings in layer
    @TODO: Check if LR is reset after its fucked up by sltr
"""

# External Lib imports
import os
from typing import List
from functools import partial

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# FastAI Imports
from fastai import text, lm_rnn

# Torch imports
import torch.nn as nn
import torch.optim as optim

# Mytorch imports
from mytorch.utils.goodies import *
from mytorch import loops, lriters as mtlr

# Local imports
import main as p2
import utils
from data import DataPuller
from options import Phase3 as params, Phase2 as p2params

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
KNOWN_DATASETS = {'imdb': 2, 'trec': 6}


'''
    Models, Data Samplers etc
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
                 n_token: int,
                 dps: list,
                 n_classes: List[int],
                 enc_wgts = None,
                 _debug=False):
        super(TextClassifier, self).__init__()
        """
        :param n_token: int representing vocab size
        :param n_classes: list representing multiple classes, each by its number of classes.
            eg. n_classes = [2] -> one task; with 2 classes
            eg. n_classes = [2, 6] -> two tasks, first with 2 classes, and one with 6.
        """

        self.device = _device

        # Load the pre-trained model
        encargs = {'ntoken': n_token, 'emb_sz': 400, 'n_hid': 1150,
                   'n_layers': 3, 'pad_token': 0, 'qrnn': False, 'bptt': 70, 'max_seq': 1400,
                   'dropouti': dps[0], 'wdrop': dps[1], 'dropoute': dps[2], 'dropouth': dps[3]}
        self.encoder = CustomEncoder(**encargs).to(self.device)

        if enc_wgts:
            self.encoder.load_state_dict(enc_wgts)

        '''
            Make multiple classifiers (depending upon n_classes)
            
            
            Explanation:
                400*3 because input is [ h_T, maxpool, meanpool ]
                50 is hidden layer dim
                2 is n_classes

                0.4, 0.1 are drops at various layers
        '''
        self.linear = [text.PoolingLinearClassifier(layers=[400 * 3, 50, cls], drops=[dps[4], 0.1]).to(self.device)
                       for cls in n_classes]
        self.domain_clf = p2.CustomLinear(layers=p2params.domclas_layers, drops=p2params.domclas_drops).to(self.device)
        self.encoder.reset()

    @property
    def layers(self):
        layers = [x for x in self.encoder.layers]
        layers += [x for linear in self.linear for x in linear.layers]
        layers += [x for x in self.domain_clf.layers]
        return torch.nn.ModuleList(layers)

    def forward(self, x: torch.tensor, domain: torch.tensor):
        """ x is sl*bs; dom is bs indicating the task. """

        # Encoding all the data
        x_proc = self.encoder(x.transpose(1, 0))

        score = []
        for dom in domain:
            score.append(self.linear[dom.item()](x_proc)[0])

        score = torch.cat(score)

        return score, x_proc

    def domain(self, x_proc):
        x_proc = list(x_proc)
        x_proc[1] = [GradReverse.apply(enc_tensr) for enc_tensr in x_proc[1]]
        return self.domain_clf(x_proc)[0]

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            predicted, _ = self.forward(x)
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
    ap.add_argument("-d", "--datasets", type=str, required=False, default="imdb,wikitext",
                    help="Comma separated two dataset names like wikitext,imdb")

    args = vars(ap.parse_args())
    QUICK = args['quick']
    DEBUG = args['debug']
    MODEL_NUM = args['modeldir']
    PRETRAINED = args['pretrained']
    MODEL_SUFFIX = args['modelsuffix']
    SAFE_MODE = args['safemode']
    UNSUP_MODEL_DIR = PATH / 'models' / str(MODEL_NUM)
    DATASETS = args['datasets'].split(',')

    assert MODEL_SUFFIX in ['_lowaux', '_hightrn', '', '_final'], 'Incorrect Suffix given with which to load model'

    params.quick = QUICK
    params.model_dir = str(UNSUP_MODEL_DIR) + ' and ' + str(MODEL_NUM)
    params.model_suffix = MODEL_SUFFIX
    params.datasets = DATASETS

    # Create representations of text using old itos
    itos_path = UNSUP_MODEL_DIR / 'itos.pkl'
    itos2 = pickle.load(itos_path.open('rb'))
    stoi2 = {v: k for k, v in enumerate(itos2)}

    data_puller = DataPuller(debug=False, max_vocab=params.max_vocab_task, min_freq=params.min_vocab_freq, trim_trn=1000, trim_val=-1)

    trn_texts_a, trn_labels_a, val_texts_a, val_labels_a, _ = data_puller.get(DATASETS[0], supervised=True,
                                                                              trim=params.quick, cached=not params.quick)
    # Lose label 2 from imdb
    if DATASETS[0] == 'imdb':
        trn_texts_a = trn_texts_a[trn_labels_a < 2]
        trn_labels_a = trn_labels_a[trn_labels_a < 2]

    # trn_texts_a = np.array([[stoi2.get(w, 0) for w in para] for para in trn_texts_a])
    # val_texts_a = np.array([[stoi2.get(w, 0) for w in para] for para in val_texts_a])

    trn_texts_b, trn_labels_b, val_texts_b, val_labels_b, itos = data_puller.get(DATASETS[1], supervised=True,
                                                                                 trim=params.quick, merge_vocab=params.max_vocab_wiki)

    if DATASETS[1] == 'imdb':
        trn_texts_b = trn_texts_b[trn_labels_b < 2]
        trn_labels_b = trn_labels_b[trn_labels_b < 2]

        # trn_texts_b = np.array([[stoi2.get(w, 0) for w in para] for para in trn_texts_b])
        # val_texts_b = np.array([[stoi2.get(w, 0) for w in para] for para in val_texts_b])

    '''
        Transform words from data_puller.itos vocabulary to that of the pretrained model (__main__.itos2)
    '''
    _itos2 = dict(enumerate(itos2))
    trn_texts_a = [[stoi2[_itos2.get(i, '_unk_')] for i in sent] for sent in trn_texts_a]
    val_texts_a = [[stoi2[_itos2.get(i, '_unk_')] for i in sent] for sent in val_texts_a]
    trn_texts_b = [[stoi2[_itos2.get(i, '_unk_')] for i in sent] for sent in trn_texts_b]
    val_texts_b = [[stoi2[_itos2.get(i, '_unk_')] for i in sent] for sent in val_texts_b]

    '''
        Make model
    '''
    dps = list(params.encoder_dropouts)
    # enc_wgts = torch.load(LM_PATH, map_location=lambda storage, loc: storage)
    enc_wgts = torch.load(UNSUP_MODEL_DIR / ('unsup_model_enc' + MODEL_SUFFIX + '.torch'), map_location=lambda storage, loc: storage)
    n_classes = [KNOWN_DATASETS[d] for d in DATASETS]
    clf = TextClassifier(device, len(itos2), dps, enc_wgts=enc_wgts if PRETRAINED else None, n_classes=n_classes)

    '''
        Setup things for training (data, loss, opt, lr schedule etc
    '''
    bs = params.bs
    loss_main_fn = torch.nn.CrossEntropyLoss()
    loss_aux_fn = torch.nn.CrossEntropyLoss()
    opt_fn = partial(optim.Adam, betas=params.adam_betas)
    opt = make_opt(clf, opt_fn, lr=0.0)
    opt.param_groups[-1]['lr'] = 0.01

    # Make data
    # @TODO: make this code compatible with one dataset (no dann)
    data_fn = partial(utils.DomainAgnosticSortishSampler, _batchsize=bs, _padidx=1)
    data_train = [{'x': trn_texts_a, 'y': trn_labels_a}, {'x': trn_texts_b, 'y': trn_labels_b}]
    data_valid = [{'x': val_texts_a, 'y': val_labels_a}, {'x': val_texts_b, 'y': val_labels_b}]
    data = {'train': data_train, 'valid': data_valid}

    # Make lr scheduler
    lr_args = {'iterations': len(data_fn(data_train)), 'cycles': 1}
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)

    save_args = {'torch_stuff': [tosave('model.torch', clf.state_dict()), tosave('model_enc.torch', clf.encoder.state_dict())]}
    save_fnames = {'torch_stuff':
                       {'hightrn':
                            {'model': 'sup_model_hightrn.torch',
                             'enc': 'sup_model_hightrn_enc.torch'},
                        'lowaux':
                            {'model': 'sup_model_lowaux.torch',
                             'enc': 'sup_model_lowaux_enc.torch'}}}

    args = {'epochs': 1, 'epoch_count': 0, 'data': data, 'device': device, 'opt': opt,
            'loss_main_fn': loss_main_fn, 'loss_aux_fn': loss_aux_fn, 'model': clf,
            'train_fn': clf, 'predict_fn': clf.predict, 'train_aux_fn': clf.domain,
            'epoch_end_hook': epoch_end_hook, 'weight_decay': params.weight_decay,
            'clip_grads_at': params.clip_grads_at, 'lr_schedule': lr_schedule,
            'loss_aux_scale': params.loss_scale if len(DATASETS) > 1 else 0,
            'data_fn': data_fn, 'eval_fn': _eval, 'eval_aux_fn': _eval,
            'save': not SAFE_MODE, 'save_params': params, 'save_dir': UNSUP_MODEL_DIR, 'save_fnames': save_fnames}

    '''
        Training schedule:
        
        1. Unfreeze one layer. Train for 1 epoch
        2 - 5. Unfreeze one layer, train for 1 epoch
        3. Train for 15 epochs (after all layers are unfrozen). Use 15 cycles for cosine annealing.
    '''
    # opt.param_groups[-1]['lr'] = 0.01
    traces = utils.dann_loop(**args)

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = utils.dann_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = utils.dann_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    opt.param_groups[-1]['lr'] = 0.01
    opt.param_groups[-2]['lr'] = 0.005
    opt.param_groups[-3]['lr'] = 0.001
    opt.param_groups[-4]['lr'] = 0.001
    lr_schedule = mtlr.LearningRateScheduler(optimizer=opt, lr_args=lr_args, lr_iterator=mtlr.CosineAnnealingLR)
    args['lr_schedule'] = lr_schedule
    args['save_above'] = np.max(traces[TRACES_FORMAT['train_acc']])
    args['epoch_count'] += 1
    traces_new = utils.dann_loop(**args)
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
    traces_new = utils.dann_loop(**args)
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

    traces_new = utils.dann_loop(**args)
    traces = [a+b for a, b in zip(traces, traces_new)]

    if not SAFE_MODE:
        mt_save(UNSUP_MODEL_DIR,
                torch_stuff=[tosave('sup_model_final.torch', clf.state_dict())],
                pickle_stuff=[tosave('final_sup_traces.pkl', traces), tosave('unsup_options.pkl', params)])
