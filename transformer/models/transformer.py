import math
import torch
import copy
from torch import nn
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder_emd = Embedder(src_vocab, d_model)
        self.encoder_pe = PositionalEncoder(d_model)
        self.encoder = Encoder(d_model, N, heads)
        
        self.decoder_emd = Embedder(trg_vocab, d_model)
        self.decoder_pe = PositionalEncoder(d_model)
        self.decoder = Decoder(d_model, N, heads)
        
        self.generator = Generator(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_output = self.encoder(self.encoder_pe(self.encoder_emd(src)), src_mask)
        d_output = self.decoder(self.decoder_pe(self.decoder_emd(trg)), e_output, src_mask, trg_mask)
        output = self.generator(d_output)
        
        return output
        
        
        
class Embedder(nn.Module):
    
    """
    vocab_size: 词典大小
    d_model:词嵌入的维度
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
        # 乘以嵌入值的目的是增大词嵌入的作用，使词嵌入与位置编码相加后原始含义不会丢失。
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout_prob=0.1, max_len=5000): 
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        
        positional_encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) #(max_len, 1)
        
        mult_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        positional_encodings[:, 0::2] = torch.sin(position * mult_term)
        positional_encodings[:, 1::2] = torch.cos(position * mult_term)
        
        positional_encodings = positional_encodings.unsqueeze(0) #(1, max_len, d_model)
        self.register_buffer('positional_encodings', positional_encodings)
    
    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        return self.dropout(x)

        
class Generator(nn.Module):
    """
    define the standard linear + softmax generator step
    """
    def __init(self, d_model, vocab_size):
        super.__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return nn.functional.log_softmax(self.linear(x), dim=-1) # 在最后一个维度上计算概率，最后一个维度长度是vocab_size
    
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super.__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, heads)) for i in range(N)])
        self.norm = LayerNorm(d_model)
    def forward(self, x, mask):
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class  Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        self.N = N
        self.layers =  nn.ModuleList([copy.deepcopy(DecoderLayer(d_model, heads)) for i in range(N)])  
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff = 2048, dropout_prob = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(heads, d_model, dropout_prob=dropout_prob)
        self.ffn = FeedForward(d_model,d_ff, dropout_prob=dropout_prob)
        self.sublayers = nn.ModuleList([SubLayer(d_model, dropout_prob) for _ in range(2)])
    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x:self.attention(x, x, x, mask))
        x = self.sublayers[1](x, self.ffn)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """
    def __init__(self, d_model, heads, d_ff = 2048, dropout_prob = 0.1):
        super().__init__()
        self.self_atten = MultiHeadAttention(heads, d_model, dropout_prob=dropout_prob)
        self.src_atten = MultiHeadAttention(heads, d_model, dropout_prob=dropout_prob)
        self.ffn = FeedForward(d_model, d_ff, dropout_prob=dropout_prob)
        self.sublayers = nn.ModuleList([SubLayer(d_model, dropout_prob) for _ in range(3)])
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_atten(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_atten(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.ffn)
        return x
                    
class SubLayer(nn.Module):
    """
    Do pre-layer normalization for input, and then run multi-head attention or feed forward,
    and finally do the residual connection.
    """
    def __init__(self, d_model, dropout_prob=0.1):
        super.__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, main_logic): # main_logic是Multi-Head Attention或者FeedForward
        x_norm = self.norm(x)
        return x + self.dropout(main_logic(x_norm))
    
class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, d_model, eps=1e-6):
        super.__init__()
        self.gama = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gama * (x-mean) / (std + self.eps) + self.beta
        
    
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):

    """
    Args:
        query: (batch_size, num_heads, seq_len_q, head_dim), given sequence that we focus on
        key: (batch_size, num_heads, seq_len_k, head_dim), the sequence to check relevance with query
        value: (batch_size, num_heads, seq_len_v, head_dim),seq_len_k == seq_len_v, usually value and key come from the same source
        mask: for encoder, mask is [batch_size, 1, 1, seq_len_k], for decoder, mask is [batch_size, 1, seq_len_q, seq_len_k]
        dropout: nn.Dropout(), optional
    Returns:
        output: (batch_size, num_heads, seq_len_q, d_v), attn: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(head_dim) #(batch_size, num_heads, seq_len_q, seq_len_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, value), scores  #(batch_size, num_heads, seq_len_q, seq_len_k) * (batch_size, num_heads, seq_len_v, head_dim) = (batch_size, num_heads, seq_len_q, head_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout_prob=0.1):
        """
        Args:
            h: number of heads
            d_model: dimension of the vector for each token in input and output
            dropout_prob: probability of dropout
        """
        super().__init__()
        self.head_dim = d_model // h
        self.num_heads = h
        # W_Q, W_K, W_V, W_O
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model), seq_len_k == seq_len_v
            mask: 
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        # 1. linear projection for query, key, value
        #    after this step, the shape of each is (batch_size, num_head, seq_len, head_dim)
        # -1维度其实就是seq_len，只是Q、K、V的seq_len可能不一样
        query, key, value = [linear(x).view(n_batches, -1, self.num_heads, self.head_dim).transpose(1,2) for linear, x in zip(self.linears, (query, key, value))]
        
        # 2. scaled dot product attention
        #    out: (batch_size, num_head, seq_len_q, head_dim) 
        out, _ = scaled_dot_product_attention(query, key, value, mask, self.dropout)
        
        # 3. "Concat" using a view and apply a final linear
        out = (
            out.transpose(1, 2).contiguous().view(n_batches, -1, self.num_heads * self.head_dim)
        )
        out = self.linears[3](out)
        
        del query, key, value
        return out
    
class FeedForward(nn.Module):
        """
        Implements FFN equation.
        x : (batch_size, seq_len_q, d_model)
        out: (batch_size, seq_len_q, d_model)
        """
        def __init__(self, d_model, d_ff, dropout_prob):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout_prob)
        def forward(self, x):
            return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))

def subsequent_mask(size):
    "Mask out subsequent positions"
    attention_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0
    

    