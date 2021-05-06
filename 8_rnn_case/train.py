import argparse
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import DataLoader
from model import RNNPredictor

# Parameters
parser = argparse.ArgumentParser(description="RNN")
parser.add_argument('--data', type=str, default='nyc_taxe',
                    help='dataset')
parser.add_argument('--filename', type=str, default='nyc_taxi.pkl',
                    help='filename')
parser.add_argument('--model', type=str, default='LSTM',
                    help='rnn type (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=True,
                    help='augment')
parser.add_argument('--input_size', type=int, default=32,
                    help='input features')
parser.add_argument('--hidden', type=int, default=32,
                    help='number of layers')
parser.add_argument('--n_layers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epoch', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cpu',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true',
                    help='save figure')
parser.add_argument('--resume', '-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')

args = parser.parse_args()
torch.manual_seed(args.seed)

# load data
time_series_data = DataLoader(data_type=args.data, filename=args.filename,
                              augment_test_data=args.augment)
train_dataset = time_series_data.batchify(args, time_series_data.X_train, args.batch_size)
test_dataset = time_series_data.batchify(args, time_series_data.X_test, args.eval_batch_size)
gen_dataset = time_series_data.batchify(args, time_series_data.X_test, 1)

# build model
feature_dim = time_series_data.X_train.size(1)
model = RNNPredictor(rnn_type=args.model,
                     encoder_input_size=feature_dim,
                     rnn_input_size=args.input_size,
                     rnn_hid_size=args.hidden,
                     decoder_output_size=feature_dim,
                     n_layers=args.n_layers,
                     dropout=args.dropout,
                     tie_weights=args.tied,
                     res_connection=args.res_connection).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
criterion = nn.MSELoss()

pass








