import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import torch

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        heads=8,
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            embed_size,
            heads,
            num_layers,
            max_length,
            src_vocab_size,
        )

        self.decoder = Decoder(
            embed_size,
            heads,
            num_layers,
            max_length,
            trg_vocab_size,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        

    def make_src_mask(self, src):
        #! This mask is necessery for handling sequences with different lengths
        #We must use it for the decoder and encoder
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        #This will be used for decoder only not to allow it to see future tokens
        N, trg_len = trg.shape
        #Ones create tensor filled with 1, trill set part above diagonal to 0 expand make it fit to our batch_size
        #This also has two roles:
        #* first is to mask unseen data.
        #* another to remove padding_indices, as they are at the end.
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
    
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)

    out = model(x, trg[:, :-1])
    print(out.shape)