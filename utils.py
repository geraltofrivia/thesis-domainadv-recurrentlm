from tqdm import tqdm
from typing import Callable, List
from mytorch.utils.goodies import *
from mytorch import lriters as mtlr


class DomainAgnosticSortishSampler:
    """
        Modified SortishSampler to handle multiple datasets.

        Sample the data so like-sized text appears together.
        Returns an iterator.

        Works well with lists and with numpy arrays.

        Domain value 0 -> data_a; 1 -> data_b

        This needs to be re-sorted every iteration.
            Hence the data is duplicated internally.
            Call reset at epoch end to resort it.

            Or you could init a new instance of this :peace:
    """

    def __init__(self, _data: list, _batchsize: int, _seqlen: int = None, _padidx=0):
        """ @TODO: snip everything with seqlen """

        _data_a, _data_b = _data

        try:
            assert len(_data_a['x']) == len(_data_a['y'])
            assert len(_data_b['x']) == len(_data_b['y'])
        except AssertionError:
            raise MismatchedDataError

        # Combine the data and maintain another domain index
        inputs, labels, domain = self._combine_domains_(src_a=_data_a, src_b=_data_b)

        self.bs = _batchsize
        self.padidx = _padidx
        self.x, self.y, self.d = self._reshuffle_(**self._sort_(inputs, labels, domain))

        self.ptr = 0

    def reset(self):
        self.x, self.y, self.d = self._reshuffle_(self.x, self.y, self.d)
        self.ptr = 0

    @staticmethod
    def _combine_domains_(src_a: dict, src_b: dict) -> (List[list], List[int], List[int]):
        """ Concatenate two datasets, and maintain a domain index"""
        inputs = list(np.concatenate([src_a['x'], src_b['x']]))
        labels = list(np.concatenate([src_a['y'], src_b['y']]))
        domain = list(np.concatenate([np.zeros(len(src_a['x'])), np.ones(len(src_b['x']))]))

        return inputs, labels, domain

    @staticmethod
    def _reshuffle_(x: list, y: list, d: list) -> (list, list, list):
        """
            Shuffles both, things inside a chunk (batch) and batches.
        :param x: list of np arr
        :param y: list of np arr
        :return: (list, list)
        """
        for i in range(len(x)):

            # Shuffle these chunks
            chunk_idx = np.random.permutation(len(x[i]))
            x[i] = x[i][chunk_idx]
            y[i] = y[i][chunk_idx]
            d[i] = d[i][chunk_idx]

        shuffle_idx = np.random.permutation(len(x))
        return [x[i] for i in shuffle_idx], [y[i] for i in shuffle_idx], [d[i] for i in shuffle_idx]

    def _sort_(self, x: List[list], y: List[int], d: List[int]) -> dict:
        """
            Create chunks of data and later will just choose from them.

                -> Sort the entire set based on lengths of x
                -> Create chunks from it
                -> Pad the chunk
                -> Add chunk to final list
        """

        idx = sorted(range(len(x)), key=lambda k: -len(x[k]))
        x, y, d = [x[i] for i in idx], [y[i] for i in idx], [d[i] for i in idx]

        final_x, final_y, final_d = [], [], []
        for ptr in range(len(x))[::self.bs]:

            # Now take a snippet of x and y based the line below
            chunk_x = x[ptr: ptr + self.bs if ptr + self.bs < len(x) else len(x)]
            chunk_y = y[ptr: ptr + self.bs if ptr + self.bs < len(x) else len(x)]
            chunk_d = d[ptr: ptr + self.bs if ptr + self.bs < len(x) else len(x)]

            # Find snippet's max len
            chunk_len = len(chunk_x[0])

            # Pad and max np arr of this batch
            npx = pad_sequence(chunk_x, chunk_len, self.padidx)
            npy = np.asarray(chunk_y)
            npd = np.asarray(chunk_d)

            # Shuffle x y & d
            chunk_idx = np.random.permutation(len(chunk_x))
            npx, npy, npd = npx[chunk_idx], npy[chunk_idx], npd[chunk_idx]

            # Append to final thing
            final_x.append(npx)
            final_y.append(npy)
            final_d.append(npd)

        return {'x': final_x, 'y': final_y, 'd': final_d}

    def __iter__(self):
        return self

    def __next__(self)->(np.ndarray, np.ndarray, np.ndarray):
        """ Iter over self.x, self.y & self.d """
        if self.ptr >= len(self.x):
            raise StopIteration
        _x, _y, _d = self.x[self.ptr], self.y[self.ptr], self.d[self.ptr]
        self.ptr += 1
        return _x, _y, _d

    def __len__(self):
        return len(self.x)


# noinspection PyCallingNonCallable,PyUnresolvedReferences
def dann_loop(epochs: int,
              data: dict,
              device: torch.device,
              opt: torch.optim,
              loss_main_fn: torch.nn,
              loss_aux_fn: torch.nn,
              model: torch.nn.Module,
              train_fn: Callable,
              predict_fn: Callable,
              train_aux_fn: Callable,
              eval_fn: Callable,
              eval_aux_fn: Callable,
              data_fn: Callable = DomainAgnosticSortishSampler,
              save: bool = False,
              save_params: dict = None,
              save_dir: Path = None,
              save_above_trn: float = -np.inf,
              save_above_aux: float = np.inf,
              save_fnames: dict = None,
              epoch_count: int = 0,
              epoch_start_hook: Callable = None,
              epoch_end_hook: Callable = None,
              batch_start_hook: Callable = None,
              batch_end_hook: Callable = None,
              weight_decay: float = 0.0,
              clip_grads_at: float = -1.0,
              lr_schedule: mtlr.LearningRateScheduler = None,
              loss_aux_scale: float = 1.0,
              notify: bool = False,
              notify_key: str = None) -> (list, list, list, list):
    """

        A generic training loop, which based on diff hook fns (defined below), should handle anything given to it.

        @TODO: Explain DANN

        The model need not be an nn.Module,
             but should have correctly wired forward and a predict function, and a auxiliary function (DANN)

        **IMPORTANT** Since DANN requires an intermediate output of the network (encoder, in most cases)
            do return it alongwith final output from the train function)

        # Data input format
            data = {
            'train': [ {'x': np arr; 'y': np arr}  # from data source 1
                       {'x': np arr; 'y': np arr}], # from data source 2
            'valid': [ {'x': np arr; 'y': np arr}  # from data source 1
                       {'x': np arr; 'y': np arr}] # from data source 2
            }

        # Saving Logic
            If the flag is enabled, give in the dir and it'll save traces and the model (and the model encoder)
                everytime training acc exceeds all prev ones.

        ## If you want to save diff parts of the model,
        Prepare save args like -
            save_args = {'torch_stuff': [tosave('model.torch', clf.state_dict()), tosave('model_enc.torch', clf.encoder.state_dict())]}
        and pass it to the model alongwith.
        If the arg is empty, it defaults to -
            save_args = {'torch_stuff': [tosave('model.torch', model.state_dict())]}

    :param epochs: number of epochs to train for
    :param data: data dict (structure specified above)
    :param device: torch device to init the tensors with
    :param opt: torch optimizer, with proper param_groups for better lr decay per laye
    :param loss_main_fn: torch.nn loss fn for the actual thing
    :param loss_aux_fn: torch.nn loss fn for the domain agnostic thing
    :param loss_aux_scale: float signifying how much to scale DANN loss with, while combining losses.
    :param model: torch module needed for
            i: grad clipping
            ii: for calling eval() and train() (regarding dropout)
    :param train_fn: a function which takes x & returns y_pred & x_proc (intermediate x)
    :param train_aux_fn: a function which takes x_proc and returns y_pred_aux (typically for dann training)
    :param predict_fn: a fn which takes x and returns y_pred
    :param eval_fn: function which when given pred and true, returns acc
    :param eval_aux_fn: function which when given pred_aux and true_aux, returns acc_aux
    :param save: [OPTIONAL] bool which wants either doesn't save, or saves at best
    :param save_dir: [OPTIONAL] Path object to which we save stuff (based on save_best)
    :param save_params: [OPTIONAL] a dict of all the params used while running and training the model.
    :param save_above_trn: [OPTIONAL] acts as threshold regarding model saving. If the current trn accuracy is less than this, won't.
    :param save_above_aux: [OPTIONAL] acts as threshold regarding model saving. If the current aux accuracy is more than this, won't.
    :param save_fnames: [OPTIONAL] reference to the model to be saved
    :param epoch_count: an int which is added with #epochs (for better representation of how many epochs have actually passed).
            You can use this for when you run the loop say 3 times, do something else and run it for another 10.
    :param epoch_start_hook: a fn that can be called @ start of every epoch (returns model, opt)
    :param epoch_end_hook: a fn that can be called @ end of every epoch (returns model, opt)
    :param batch_start_hook: a fn that can be called @ start of every batch (returns model, opt)
    :param batch_end_hook: a fn that can be called @ end of every batch (returns model, opt)
    :param weight_decay: a L2 ratio (as mentioned in (https://arxiv.org/pdf/1711.05101.pdf)
    :param clip_grads_at: in case you want gradients clipped, send the max val here
    :param lr_schedule: a schedule that is called @ every batch start.
    :param data_fn: a class to which we can pass X and Y, and get an iterator.
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

    assert (not save) or (save and save_dir), "No save dir specified."

    # Epoch level
    for e in range(epoch_count, epochs + epoch_count):

        per_epoch_loss_main = []
        per_epoch_loss_aux = []
        per_epoch_tr_acc_main = []
        per_epoch_tr_acc_aux = []

        # Train
        with Timer() as timer:

            # Enable dropouts
            model.train()

            # @TODO: Add hook at start of epoch (how to decide what goes in)
            if epoch_start_hook:
                epoch_start_hook()

            # Make data
            trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])

            for x, y, y_aux in tqdm(trn_dl):

                if batch_start_hook:
                    batch_start_hook()

                opt.zero_grad()

                if lr_schedule:
                    lrs.append(update_lr(opt, lr_schedule.get()))

                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)
                _y_aux = torch.tensor(y_aux, dtype=torch.long, device=device)

                # A. Regular stuff
                y_pred, x_proc = train_fn(_x, _y_aux)
                loss_main = loss_main_fn(y_pred, _y)

                # B. Aux stuff
                y_pred_aux = train_aux_fn(x_proc)
                loss_aux = loss_aux_fn(y_pred_aux, _y_aux)

                # C. Add losses with scale.
                loss = loss_main + (loss_aux_scale * loss_aux)

                # Logging
                per_epoch_tr_acc_main.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_tr_acc_aux.append(eval_aux_fn(y_pred=y_pred_aux, y_true=_y_aux).item())
                per_epoch_loss_main.append(loss_main.item())
                per_epoch_loss_aux.append(loss_aux.item())

                # Compute gradients
                loss.backward(retain_graph=False)

                if clip_grads_at > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads_at)
                for group in opt.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-weight_decay * group['lr'], param.data)

                opt.step()
                if batch_end_hook:
                    batch_end_hook()

            if epoch_end_hook:
                epoch_end_hook()

        # Val
        with torch.no_grad():

            # Disable dropouts
            model.eval()

            per_epoch_vl_acc = []
            for x, y, y_aux in tqdm(val_dl):
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)
                _y_aux = torch.tensor(y_aux, dtype=torch.long, device=device)

                y_pred = predict_fn(_x, _y_aux)[0]

                per_epoch_vl_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())

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
              % {'epo': e + epoch_count,
                 'loss': float(np.mean(per_epoch_loss_main)),
                 'loss_aux': float(np.mean(per_epoch_loss_aux)),
                 'tracc': float(np.mean(per_epoch_tr_acc_main)),
                 'tracc_aux': float(np.mean(per_epoch_tr_acc_aux)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

        # Save block (flag and condition)
        if save and train_acc_main[-1] >= save_above_trn:

            # Update threshold
            save_above_trn = train_acc_main[-1]

            # Adding epoch info along with options
            if save_params:
                save_params.epoch = e
            else:
                save_params = {'epoch': e}

            # Prepare save_args if none
            if not save_fnames:
                save_fnames = {'torch_stuff':
                                   {'hightrn':
                                        {'model': 'model_hightrn.torch',
                                         'enc': 'model_hightrn_enc.torch'},
                                    'lowaux':
                                        {'model': 'model_lowaux.torch',
                                         'enc': 'model_lowaux_enc.torch'}}}

            # Call save function and save
            mt_save(save_dir,
                    torch_stuff=[tosave(save_fnames['torch_stuff']['hightrn']['model'], model.state_dict()),
                                 tosave(save_fnames['torch_stuff']['hightrn']['enc'], model.encoder.state_dict())],
                    pickle_stuff=[
                        tosave('traces.pkl', [train_acc_main, val_acc, train_acc_aux, train_loss_main, train_loss_aux, lrs]),
                        tosave('options.pkl', save_params)])

            print(f"Model saved on Epoch {e} at {save_dir} because of highest training acc so far")

            # Log the saved thing
            saved_info['epoch'] = e
            saved_info['accuracy'] = val_acc[-1]
            saved_info['directory'] = save_dir

        # Save block (flag and condition) (When more than 2 epochs have passed and we see lowest aux accuracy)
        if save and e > 1 and train_acc_aux[-1] <= save_above_aux:

                # Update threshold
                save_above_aux = train_acc_aux[-1]

                # Adding epoch info along with options
                if save_params:
                    save_params['epoch'] = e
                else:
                    save_params = {'epoch': e}

                # Prepare save_args if none
                if not save_fnames:
                    save_fnames = {'torch_stuff':
                                       {'hightrn':
                                            {'model': 'model_hightrn.torch',
                                             'enc': 'model_hightrn_enc.torch'},
                                        'lowaux':
                                            {'model': 'model_lowaux.torch',
                                             'enc': 'model_lowaux_enc.torch'}}}

                # Call save function and save
                mt_save(save_dir,
                        torch_stuff=[tosave(save_fnames['torch_stuff']['lowaux']['model'], model.state_dict()),
                                     tosave(save_fnames['torch_stuff']['lowaux']['enc'], model.encoder.state_dict())],
                        pickle_stuff=[
                            tosave('traces.pkl', [train_acc_main, val_acc, train_acc_aux, train_loss_main, train_loss_aux, lrs]),
                            tosave('options.pkl', save_params)])

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


TRACES_FORMAT = {name: i for i, name in enumerate(['train_acc_main', 'val_acc', 'train_acc_aux', 'train_loss_main', 'train_loss_aux', 'lrs'])}