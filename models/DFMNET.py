import torch
import torch.nn as nn

class DFMNET(nn.Module):
    def __init__(self, n_input:int, n_output:int, n_rnn_layer:int=2, n_rnn_hidden:int=256, n_kdn_hidden:int = 256):
        super(DFMNET, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_rnn_layer=n_rnn_layer
        self.n_rnn_hidden = n_rnn_hidden
        self.n_kdn_hidden = n_kdn_hidden
        self.kdn_drop_out_rate = 0.2

        self._buildModel()

    def _buildModel(self):
        self.SEN = nn.GRU(
            input_size=self.n_input,
            hidden_size=self.n_rnn_hidden,
            num_layers=self.n_rnn_layer,
            dropout=0.5
        )

        self.KDN = nn.Sequential(
            nn.Linear(self.n_input + self.n_rnn_hidden , self.n_kdn_hidden),
            nn.ReLU(),

            nn.Linear(self.n_kdn_hidden, self.n_kdn_hidden),
            nn.ReLU(),

            nn.Linear(self.n_kdn_hidden, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        x = input.transpose(0, 1)
        h, _ = self.SEN(x)
        r = torch.cat([h[-1, :, :], input[:, -1, :]], 1)
        return self.KDN(r)