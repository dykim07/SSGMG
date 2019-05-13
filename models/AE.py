import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, n_input:int, n_latent_dim:int):
        super(AE, self).__init__()
        self.n_input = n_input
        self.n_latent_dim = n_latent_dim
        self.n_hidden = 128
        self._buildModel()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def _buildModel(self):

        self.encoder = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent_dim, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_input)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)


    def forward(self, input):
        r = self.encoder(input)
        return self.decoder(r)

    def transform(self, input):
        return self.encoder(input)

    def inverse_transform(self, input):
        return self.decoder(input)


class AEVector(nn.Module):
    def __init__(self, n_input:int):
        super(AEVector, self).__init__()
        self.n_input = n_input
        self.n_hidden = 128
        self.__build()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def __build(self):
        self.encoder1 = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden)

        )

        self.encoder_split = nn.Linear(
            self.n_hidden , self.n_hidden *2
        )

        self.encoder21 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 2)
        )

        self.encoder22 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 1)
        )

        self.decoder11 = nn.Sequential(
            nn.Linear(2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden)
        )

        self.decoder12 = nn.Sequential(
            nn.Linear(1, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden)
        )

        self.decoder_merge = nn.Linear(
            self.n_hidden * 2, self.n_hidden
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_input)
        )

    def forward(self, input):
        l, v = self.transform(input)
        return self.inverse_transform(l, v), l, v

    def transform(self, input ):
        r = self.encoder1(input)
        r = self.encoder_split(r)

        l = self.encoder21(r[:, :int(r.size(-1)/2)])
        v = self.encoder22(r[:, int(r.size(-1)/2):])

        return l, v

    def inverse_transform(self, l, v):
        l1 = self.decoder11(l)
        l2 = self.decoder12(v)

        l = torch.cat([l1, l2], 1)
        l = self.decoder_merge(l)

        return self.decoder2(l)

