from turtle import position
import torch.nn as nn
import torch
from better_trans_block import TransformerBlock

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, emb_dim, n_heads, n_blocks, max_seq_length, input_vocab_size):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_length = max_seq_length
        self.input_vocab_size = input_vocab_size

        self.word_embeddings = nn.Embedding(input_vocab_size, emb_dim) #So this will learn embeddings for each word in our vocabulary
        self.positional_embeddings = nn.Embedding(max_seq_length, emb_dim) # This will learn embeddings for each possible position in the sequence
        

        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_dim, n_heads) for _ in range(n_blocks)])

        # In that case this is all as we will just pass embeddings to the decoder

    def forward(self, X, src_mask):
        #as an input we expect it to have size batch x seq len and be in a form [1,3,20,50, 65]
        batch, seq_len = X.shape
        word_embeddings = self.word_embeddings(X)
        
        positional_embeddings = torch.arange(seq_len) # I need that much embeddings, if seq_len > max_seq_length we are in trouble
        #now positional_embeddings have shape seq_len, but we need the same ebedding for everything in batch
        positional_embeddings = positional_embeddings.expand(batch, seq_len) #now it's batch_size x seq_len
        positional_embeddings = self.positional_embeddings(positional_embeddings) #finally batch_size x seq_len x emb_dim

        embeddings = word_embeddings + positional_embeddings # they must match the dimensions

        for block in self.transformer_blocks:
            embeddings = block(embeddings, embeddings, embeddings, src_mask)

        #In that case we are not using encoder for any classification so we just return embeddings
        return embeddings


if __name__ == "__main__":
    
    sequence = torch.tensor([[10,20,30,0,0], [8,1,2,143,99]])
    src_pad_idx = 0#123
    src_mask = (sequence != src_pad_idx).unsqueeze(1).unsqueeze(2)

    print(src_mask)
    encoder = Encoder(5, 8, 6, 1024, 2048)

    embeddings = encoder(sequence, src_mask)
    print(embeddings.shape)
    print(embeddings)




