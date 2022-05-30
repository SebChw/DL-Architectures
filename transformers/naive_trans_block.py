import torch.nn as nn
import torch

class Sc_Dot_Pro_Attention(nn.Module):
    def __init__(self, att_dim, embedding_dim):
        super(Sc_Dot_Pro_Attention, self).__init__()
        self.att_dim = att_dim
        self.embedding_dim = embedding_dim
        #Input will have dimensionality sequence_length x embedding_dim
        #All matrices producing query, keys, values will be embeding_dim x att_dim
        #The product will produce sequence_length x att_dim. Remember that we will have 6 attention heads
        #So after conncatenation dimensions will match. Then we can make residual connection
        self.Q = nn.Linear(embedding_dim, att_dim)
        self.K = nn.Linear(embedding_dim, att_dim)
        self.V = nn.Linear(embedding_dim, att_dim)

    def forward(self, X):
        #Also take into account that all this will be batched
        queries = self.Q(X)
        keys = self.K(X)
        values = self.V(X)

        attention_scores = queries @ keys.permute(0,2,1) # (seq_len x att_dim ) @ (att_dim x seq_len) -> (seq_len x seq_len)
        attention_scores_scaled = attention_scores / (self.att_dim ** (1/2))
        attention_weights = torch.softmax(attention_scores_scaled, dim=2) # Here it is very important to not mess up the dimension along we do this

        context = attention_weights @ values # (seq_len x seq_len) @ (seq_len x att_dim) -> (seq_len x att_dim)
        
        return context 
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.att_dim = self.embedding_dim // self.n_heads #integer division

        assert self.att_dim * self.n_heads == self.embedding_dim, "Embed size must by integer divisible by number of heads"

        self.heads = nn.ModuleList([Sc_Dot_Pro_Attention(self.att_dim, self.embedding_dim) for i in range(n_heads)])

    def forward(self, X):
        output = []
        for head in self.heads:
            output.append(head(X))
        
        return torch.cat(output, dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embedding_dim):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim

        self.attention = MultiHeadAttention(n_heads, embedding_dim)
        self.att_norm = nn.LayerNorm(self.embedding_dim)

        self.expansion_factor = 4
        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, self.expansion_factor*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.expansion_factor*self.embedding_dim, self.embedding_dim)
        )

        self.linear_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, X):
        X_att = self.attention(X)
        X = self.att_norm(X_att + X)

        X_lin = self.linear(X)
        X = self.linear_norm(X_lin + X)

        return X

class Transformer():
    def __init__(self, model_dim = 512):
        self.model_dim = model_dim


if __name__ == "__main__":
    single_att = Sc_Dot_Pro_Attention(64,512)
    inp = torch.randn(64, 32, 512) # batch_size = 64, seq_length = 32, embed_dim = 512
    
    out = single_att(inp)
    print(out.shape)

    multi_head_att = MultiHeadAttention(n_heads = 4, embedding_dim = 512)

    inp = torch.randn(64, 32, 512) # batch_size = 64, seq_length = 32, embed_dim = 512
    
    out = multi_head_att(inp)
    print(out.shape)

    trans_block = TransformerBlock(n_heads= 4, embedding_dim = 512,)
    out = trans_block(inp)
    print(out.shape)