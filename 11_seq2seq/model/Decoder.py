import torch.nn as nn
import random
from torch.autograd import Variable
import torch


class VanillaDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length,
                       teacher_forcing_ratio, sos_id, use_cuda):
        super(VanillaDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
        self.use_cuda = use_cuda

    def forward_step(self, inputs, hidden):
        batch_size = inputs.size(1)
        embedded = self.embedding(inputs)
        embedded.view(1, batch_size, self.hidden_size)
        rnn_output, hidden = self.gru(embedded, hidden)
        rnn_output = rnn_output.squeeze(0)
        output = self.log_softmax(self.out(rnn_output))

        return output, hidden

    def forward(self, context_vector, targets):
        target_vars, target_lengths = targets
        batch_size = context_vector.size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id]*batch_size]))

        decoder_hidden = context_vector

        max_target_length = max(target_lengths)
        decoder_output = Variable(torch.zeros(
            max_target_length,
            batch_size,
            self.output_size
        ))

        use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        for t in range(max_target_length):
            decoder_output_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_output[t] = decoder_output_on_t
            if use_teacher_forcing:
                decoder_input = target_vars[t].unsqueeze(0)
            else:
                decoder_input = self._decode_to_index(decoder_output_on_t)

        return decoder_output, decoder_hidden

    def evaluate(self, context_vector):
        batch_size = context_vector.size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id]*batch_size]))
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.max_length,
            batch_size,
            self.output_size
        ))

        for t in range(self.max_length):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            decoder_input = self._decode_to_index(decoder_outputs_on_t)

        return self._decode_to_indices(decoder_outputs)

    def _decode_to_index(self, decoder_output):
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0, 1)
        return index

    def _decode_to_indices(self, decoder_outputs):
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)

        for b in range(batch_size):
            top_ids = self._decode_to_index(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0].cpu().numpy())

        return decoded_indices