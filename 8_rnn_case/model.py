import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path
import shutil
from torch.autograd import Variable


class RNNPredictor(nn.Module):
    def __init__(self, rnn_type, encoder_input_size, rnn_input_size,
                 rnn_hid_size, decoder_output_size, n_layers, dropout=0.5,
                 tie_weights=False, res_connection=False):
        super(RNNPredictor, self).__init__()
        self.encoder_input_size = encoder_input_size
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Linear(encoder_input_size, rnn_input_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_input_size, rnn_hid_size, n_layers, dropout=dropout)
        else:
            raise ValueError("Wrong option!")

        self.decoder = nn.Linear(rnn_hid_size, decoder_output_size)

        if tie_weights:
            if rnn_hid_size != rnn_input_size:
                raise ValueError("tie weights should have same shape!")
            self.decoder.weight = self.encoder.weight

        self.res_connection = res_connection
        self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.n_layers = n_layers

    def init_weights(self):
        nn.init.xavier_uniform(self.encoder.weight)
        nn.init.xavier_uniform(self.decoder.weight)
        nn.init.xavier_uniform(self.rnn.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, input, hidden, return_hiddens=False, noise=False):
        # (seq_lenth * batch, feature_dim)
        embedding = self.dropout(self.encoder(input.contiguous().view(-1, self.encoder_input_size)))
        # (seq_lenth, batch, feature_dim)
        embedding = embedding.view(-1, input.size(1), self.rnn_hid_size)

        output, hidden = self.rnn(embedding, hidden)

        output = self.dropout(output)
        decoder = self.decoder(output.view(output.size(0)*output(1), output.size(2)))
        decoder = decoder.view(output.size(0), output.size(1), output.size(2))
        if self.res_connection:
            decoder += input

        return decoder, hidden

    def save_checkpoint(self, state, is_best):
        args = state['args']
        checkpoint_dir = Path('save', args.data, 'checkpoint')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')
        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save', args.data, 'model_best')
            model_best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

    def initialize(self, args, feature_dim):
        self.__init__(rnn_type=args.model,
                      encoder_input_size=feature_dim,
                      rnn_input_size=args.input_size,
                      rnn_hid_size=args.hidden,
                      decoder_output_size=feature_dim,
                      n_layers=args.n_layers,
                      dropout=args.dropout,
                      tie_weights=args.tied,
                      res_connection=args.res_connection)
        self.to(args.device)

    def load_checkpoint(self, args, checkpoint, feature_dim):
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size = args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss

