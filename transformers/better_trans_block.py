import torch.nn as nn
import torch


#Linear layer by default apply linear transformation only to the last dimension
class Attention(nn.Module):
    """It's called an attention as it may be either typical attention, self attention or masked attention depending on what you pass to it"""
    def __init__(self, emb_dim, heads=8):
        super(Attention, self).__init__()


        self.emb_dim = emb_dim
        self.heads = heads

        #We will compute query, key, value for every attention head using one linear operation
        self.Q = nn.Linear(self.emb_dim, self.heads*self.emb_dim)
        self.K = nn.Linear(self.emb_dim, self.heads*self.emb_dim)
        self.V = nn.Linear(self.emb_dim, self.heads*self.emb_dim)

        self.reduce_heads = nn.Linear(heads * self.emb_dim, self.emb_dim)

    def forward(self, pot_values, pot_keys, pot_query, mask):
        #Here pot means potential, to indicate that these are not yet queries keys etc.
        N = pot_values.shape[0] # Batch size

        #set lengths for the keys and values must be the same, as for every key we have one corresponding value
        #But set length for the query may be different.
        #Not for self attention but for general it may happen
        value_key_t = pot_values.shape[1]
        query_t = pot_query.shape[1]

        #So we at first calculate one big matrix multiply for all heads
        #And then we split this data so for each weight we calculate similarities separately
        queries = self.Q(pot_query).view(N, query_t, self.heads, self.emb_dim)
        keys = self.K(pot_keys).view(N, value_key_t, self.heads, self.emb_dim)
        values = self.V(pot_values).view(N, value_key_t, self.heads, self.emb_dim)

        #This magic is explained here very well
        #https://rockt.github.io/2018/04/30/einsum
        similarity_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        #thanks to computing similarities between different features separately they won't cancel out.
        #this will have shape (batch_size, num_of_heads, num_of_queries, num_of_keys)
        if mask is not None:
            # this will basically set scores to 0
            # So if we pass boolean mask tensor then we have masked attention 
            similarity_scores = similarity_scores.masked_fill(mask == 0, float("-inf"))
        
        #Dim 3 is number of keys. So for each query we have normalized row
        #We additionally scale similarities
        attention =  torch.softmax(similarity_scores / (self.emb_dim ** (1/2)),dim=3)

        #Again magic happens with eigensums. Here we just calculate linear combinations of values
        #reshaping works like concatenation here.
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_t, self.heads * self.emb_dim
        )
        #this will have shape (batch_size, num_of_queries, concatenated_head_out_dim) -> for every batch for every query we created a context vector measuring similarities between them.

        #At the very end I want to make it emb dim again for the residual connection to work.
        out = self.reduce_heads(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, expanse_factor = 4, dropout_p = 0):
        super(TransformerBlock, self).__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.expanse_factor = expanse_factor

        self.attention = Attention(emb_dim = emb_dim, heads = n_heads)
        self.layer_norm_att = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, expanse_factor * emb_dim),
            nn.ReLU(),
            nn.Linear(expanse_factor * emb_dim, emb_dim)
        )
        self.layer_norm_mlp = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self,  pot_values, pot_keys, pot_query, mask):
        attention = self.attention(pot_values, pot_keys, pot_query, mask)

        norm_attention = self.dropout(self.layer_norm_att(attention +  pot_query)) #3 things happens. Skip connection between input and output, layer norm and dropout
        mlp_out = self.mlp(norm_attention) # this norm_attention has 3 dimensions (batch, query_len, emb_dim). However as we read in documentation this is fine for linear layer
        norm_mlp_out = self.dropout(self.layer_norm_mlp(mlp_out + norm_attention))

        return norm_mlp_out

if __name__ == "__main__":
    single_att = Attention(512,8)
    inp = torch.randn(64, 32, 512) # batch_size = 64, seq_length = 32, embed_dim = 512
    
    out = single_att(inp, inp, inp, None)
    print(out.shape)
