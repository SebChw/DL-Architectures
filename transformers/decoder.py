from turtle import position
import torch.nn as nn
import torch
from better_trans_block import TransformerBlock
from better_trans_block import Attention

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(DecoderBlock, self).__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.attention = Attention(emb_dim, n_heads)
        self.transf_block = TransformerBlock(emb_dim, n_heads)

    def forward(self, X, encoder_values, encoder_keys, src_mask, trg_mask):

        self_attention = self.attention(X, X, X, trg_mask)
        self_attention = self.layer_norm(self_attention + X) # Now this serves as a query to the transformer block
        
        return self.transf_block(encoder_values, encoder_keys, self_attention, src_mask)

        

class Decoder(nn.Module):
    def __init__(self, emb_dim, n_heads, n_blocks, max_seq_length, target_vocab_size):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_length = max_seq_length
        self.target_vocab_size = target_vocab_size

        self.word_embeddings = nn.Embedding(target_vocab_size, emb_dim) #So this will learn embeddings for each word in our vocabulary
        self.positional_embeddings = nn.Embedding(max_seq_length, emb_dim) # This will learn embeddings for each possible position in the sequence
        

        self.transformer_blocks = nn.ModuleList([DecoderBlock(emb_dim, n_heads) for _ in range(n_blocks)])

        #Here we will use decoder's output for classification that's why we need to pass embeddings through linear layer to have correct shape
        self.embed_to_logits = nn.Linear(emb_dim, target_vocab_size)

    def forward(self, X, encoder, src_mask, target_mask):
        #as an input we expect it to have size batch x seq len and be in a form [1,3,20,50, 65]
        batch, seq_len = X.shape
        word_embeddings = self.word_embeddings(X)
        
        positional_embeddings = torch.arange(seq_len) # I need that much embeddings, if seq_len > max_seq_length we are in trouble
        #now positional_embeddings have shape seq_len, but we need the same ebedding for everything in batch
        positional_embeddings = positional_embeddings.expand(batch, seq_len) #now it's batch_size x seq_len
        positional_embeddings = self.positional_embeddings(positional_embeddings) #finally batch_size x seq_len x emb_dim

        embeddings = word_embeddings + positional_embeddings # they must match the dimensions

        for block in self.transformer_blocks:
            embeddings = block(embeddings, encoder, encoder, src_mask, target_mask)

        return self.embed_to_logits(embeddings) # From this we can perform sampling the most probable word



