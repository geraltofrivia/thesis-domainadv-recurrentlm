from mytorch.utils.goodies import *


class Phase2:
    lr = FancyDict(
        sltr_cutfrac=0.1,
        sltr_ratio=32,
        init=0.003,
        dscr=1.3,
    )
    adam_betas = (0.8, 0.99)
    weight_decay = 0.0
    clip_grads_at = -1.0
    max_vocab_task = 60000
    max_vocab_wiki = 1000
    min_vocab_freq = 2

    domclas_layers = [400*3, 50, 2]
    phase1=True

    # Dropouts
    encoder_drops = list(np.asarray([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7)
    decoder_drops = 0.1 * 0.7
    domclas_drops = [0.2, 0.1]

    bs = 20
    loss_scale = 0.5

class Phase3:
    encoder_dropouts = np.asarray([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5
    bs = 24
    adam_betas = (0.7, 0.99)
    clip_grads_at = 0.3
    weight_decay = 1e-7
    max_vocab_task = 70000
    max_vocab_wiki = 2000
    min_vocab_freq = 2


class _Phase2:
    # Collected (overwrote P2) with defaults from phase2.py
    lr = FancyDict(
        sltr_cutfrac=0.1,
        sltr_ratio=32,
        init=0.003,
        dscr=1.3,
    )
    adam_betas = (0.8, 0.99)
    weight_decay = 0.0
    clip_grads_at = -1.0
    max_vocab_task = 60000
    max_vocab_wiki = 0

    domclas_layers = [400*3, 50, 2]
    phase1=True

    # Dropouts
    encoder_drops = list(np.asarray([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7)
    decoder_drops = 0.1 * 0.7
    domclas_drops = [0.2, 0.1]

    bs = 24
    loss_scale = 0.5
