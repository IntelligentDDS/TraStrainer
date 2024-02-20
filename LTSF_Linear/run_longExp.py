import torch
import random
import numpy as np
from LTSF_Linear.exp_main import Exp_Main

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class Argument:
    def __init__(self, input, model_id='weather'):
        # basic config
        self.model_id = model_id
        self.model = 'DLinear'
        self.input = input

        # data loader
        self.data = 'custom'
        self.features = 'M'  # multivariate predict multivariate
        self.target = 'OT'
        self.freq = 'h'  # hourly
        self.checkpoints = './checkpoints/'

        # forecasting task
        self.seq_len = 96  # input sequence length
        self.label_len = 0  # start token length
        self.pred_len = 1  # prediction sequence length

        # DLinear
        self.individual = False

        # Formers
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.05
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = True
        self.do_predict = True

        # optimization
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 100
        self.batch_size = 4
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'Exp'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False


def metric_learner(args):
    exp = Exp_Main(args)
    model = exp.train()
    torch.cuda.empty_cache()
    return model


def metric_infer(args, model=None):
    exp = Exp_Main(args)
    preds, trues = exp.test(model=model)
    torch.cuda.empty_cache()
    return preds, trues
