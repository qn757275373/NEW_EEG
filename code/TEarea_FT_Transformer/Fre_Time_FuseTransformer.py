import torch
from torch import nn
import math
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import scipy.io as sci
import datetime
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
# import torch.nn.functional as F
from BAM import *
from configure import opt

# Transformer Parameters
channel_size = 128
time_d_model = 128  # Time Embedding Size
d_model = 128  # Embedding Size
d_ff = 2048  # FeedForward dimension (62-256-62线性提取的过程)
# d_ff = 256
d_k = d_v = 128 # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 32


# 对decoder的输入来屏蔽未来信息的影响，这里主要构造一个矩阵
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    # 创建一个三维矩阵
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # triu产生上三角矩阵，k=1对角线的位置上移一个，k=0表示正常的上三角，k=-1表示对角线位置下移1个对角线
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    # byte转换成只有0和1，上三角全是1，下三角全是0
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # subsequence_mask = torch.from_numpy(subsequence_mask).byte().cuda()
    # subsequence_mask = subsequence_mask.data.eq(0)  # 0-True,1-False
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# Attention过程
# 通过 Q 和 K 计算出 scores，然后将 scores 和 V 相乘，得到每个单词的 context vector
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask为True的位置全部填充为-无穷，无效区域不参与softmax计算
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


# 多头Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                    2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        # 转换d_model维度
        output = self.fc(context)  # [batch_size, len_q, d_model]
        d_model = output.shape[2]
        # 残差连接+LN
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU()
        )
        # Swish激活函数
        # Swish激活函数
        # self.swish = My_Swish()
        self.batchNorm = nn.BatchNorm1d(d_ff)

        self.fc2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs

        input_fc1 = self.fc(inputs)

        # Swish激活函数
        # input_fc1_sw = self.swish(input_fc1)

        # (b, t, c) - > (b, c, t)
        # input_fc1_sw = input_fc1_sw.permute(0, 2, 1)
        input_fc1_sw = input_fc1.permute(0, 2, 1)
        input_bn = self.batchNorm(input_fc1_sw)
        # (b, t, c) - > (b, c, t)
        input_bn = input_bn.permute(0, 2, 1)

        output = self.fc2(input_bn)
        d_model = output.shape[2]
        # output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
#         # return nn.BatchNorm1d(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]






class FT_Fuse_EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(FT_Fuse_EncoderLayer, self).__init__()
        self.Fre_self_attn = MultiHeadAttention(d_model)
        self.Time_self_attn = MultiHeadAttention(d_model)
        self.Fusion_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)
        self.linear = nn.Linear(d_model*2, d_model)

    def forward(self, Fre_pos_inputs,Time_pos_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        Fre_outputs, attn = self.Fre_self_attn(Fre_pos_inputs, Fre_pos_inputs, Fre_pos_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        Time_outputs, attn = self.Time_self_attn(Time_pos_inputs, Time_pos_inputs, Time_pos_inputs,
                                               enc_self_attn_mask)  #
        # enc_outputs = torch.cat((Time_outputs[:,:4,:], Fre_outputs[:,:4,:]), dim=1)
        # enc_outputs = torch.cat((Time_outputs, Fre_outputs), dim=1)
        # FFT Features
        # enc_outputs += fourier_transform(enc_outputs)
        # enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        # token fusion cross-attention
        Fre_outputs = self.pos_ffn(Fre_outputs)
        Time_outputs = self.pos_ffn(Time_outputs)
        Fre_tokens =  Fre_outputs[:,:4,:]
        Time_tokens = Time_outputs[:,:4,:]
        fusion_token1 = torch.cat((Time_tokens,Fre_tokens), dim=1)
        fusion_token2 = torch.cat((Time_tokens,Fre_tokens), dim=-1)
        fusion_token2 = self.linear(fusion_token2)
        # fusion_ouput, attn = self.Fusion_attn(fusion_token2,fusion_token1,fusion_token1,enc_self_attn_mask)
        fusion_ouput = torch.cat((fusion_token1,fusion_token2), dim=1)
        enc_outputs = self.pos_ffn(fusion_ouput)
        return enc_outputs, attn


class FT_BAM_Fuse_FuseEncoder(nn.Module):
    def __init__(self):
        super(FT_BAM_Fuse_FuseEncoder, self).__init__()
        self.Fre_src_emb = nn.Linear(1, d_model, bias=False)
        self.Fre_pos_emb = nn.Parameter(torch.randn(1, 668, d_model))
        self.Fre_cls_token = nn.Parameter(torch.randn(1, 8, d_model))
        self.Time_cls_token = nn.Parameter(torch.randn(1, 8, d_model))
        self.Time_src_emb = nn.Linear(128, time_d_model, bias=False)
        self.Time_pos_emb = nn.Parameter(torch.randn(1, 448, time_d_model))
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([FT_Fuse_EncoderLayer(d_model=d_model) for _ in range(n_layers)])
        self.my_bam = BAM(d_model)
    def forward(self, Fre_inputs, Time_inputs):
        Fre_inputs = Fre_inputs.to(torch.float32)
        Time_inputs = Time_inputs.to(torch.float32)
        # Fre
        Fre_inputs = self.Fre_src_emb(Fre_inputs)
        # add class_token
        b, n, _ = Fre_inputs.shape
        # cls_tokens = repeat(self.Fre_cls_token, '() n e -> b n e', b=b)
        # Fre_inputs = torch.cat((cls_tokens, Fre_inputs), dim=1)
        # Position embedding
        # Fre_inputs += self.Fre_pos_emb[:, :(n + 4)]
        # Fre_inputs += self.Fre_pos_emb[:, :n]
        Fre_inputs_bam = rearrange(Fre_inputs, 'b f c -> b c f')
        Fre_inputs_bam = Fre_inputs_bam.unsqueeze(2)
        BAM_Fre_outputs = self.my_bam(Fre_inputs_bam)
        BAM_Fre_outputs = BAM_Fre_outputs.squeeze(2)
        BAM_Fre_outputs = rearrange(BAM_Fre_outputs, 'b c f -> b f c')
        Fre_inputs = Fre_inputs + BAM_Fre_outputs
        Fre_pos_inputs = self.dropout(Fre_inputs)
        # Time
        b, n, _ = Time_inputs.shape
        Time_inputs = self.Time_src_emb(Time_inputs)
        cls_tokens = repeat(self.Time_cls_token, '() n e -> b n e', b=b)
        Time_inputs = torch.cat((cls_tokens, Time_inputs), dim=1)
        # Time_inputs += self.Time_pos_emb[:, :(n+4)]
        Time_inputs_bam = rearrange(Time_inputs, 'b t c -> b c t')
        Time_inputs_bam = Time_inputs_bam.unsqueeze(2)
        Time_inputs_bam = self.my_bam(Time_inputs_bam)
        Time_inputs_bam = Time_inputs_bam.squeeze(2)
        Time_inputs_bam = rearrange(Time_inputs_bam, 'b c t -> b t c')
        Time_inputs = Time_inputs + Time_inputs_bam
        Time_pos_inputs = self.dropout(Time_inputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        # enc_outputs = torch.cat((Fre_pos_inputs, Time_pos_inputs), dim=1)

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(Fre_pos_inputs, Time_pos_inputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class channel_attention(nn.Module):
    def __init__(self, sequence_num=400, channel_size = 128, inter=10):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            nn.LayerNorm(channel_size),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(channel_size),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(channel_size, channel_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(channel_size),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))
        self.drop_out_last = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')

        # add layerNorm
        temp = nn.LayerNorm(channel_size).cuda()(temp)

        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        # ====================save channel_atten_score
        if(opt.current_epoch == 1 or opt.current_epoch % 50 == 0):
            np.save("/home/mly/PycharmProjects/weizhan-eeg/egg/code/TEarea_FT_Transformer/beforF_channelAttention_map/attention_map_time_" + '%d' % opt.current_epoch + ".npy",
                    channel_atten_score[0, :, :, :].detach().cpu().numpy())
        # ====================end save channel_atten_score
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')

        # Add: dropout + residual block
        out = self.drop_out_last(out)
        res = out + x
        return res

class FT_Fuse_FuseEncoder(nn.Module):
    def __init__(self):
        super(FT_Fuse_FuseEncoder, self).__init__()
        self.Fre_src_emb = nn.Linear(1, d_model, bias=False)
        # self.Fre_pos_emb = nn.Parameter(torch.randn(1, 660, d_model))
        self.Fre_pos_emb = nn.Parameter(torch.randn(1, 812, d_model))
        self.Fre_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.Time_cls_token = nn.Parameter(torch.randn(1, 4, d_model))
        self.Time_src_emb = nn.Linear(128, time_d_model, bias=False)
        self.Time_pos_emb = nn.Parameter(torch.randn(1, 488, time_d_model))
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([FT_Fuse_EncoderLayer(d_model=d_model) for _ in range(n_layers)])
        self.my_bam = BAM(d_model)
        self.F_channel_atten = channel_attention(5, inter=2)
        self.T_channel_atten = channel_attention(400)

    def forward(self, Fre_inputs, Time_inputs):
        Fre_inputs = Fre_inputs.to(torch.float32)
        Fre_inputs_tmp = Fre_inputs.to(torch.float32)
        Time_inputs = Time_inputs.to(torch.float32)
        # ================channel attention
        # b, 5, c -> b, 1, c, 5 -> b 5 c -> b 5*c+20 1
        Fre_inputs = Fre_inputs[:,:5, :].permute(0, 2, 1).unsqueeze(dim=1)
        # Fre_inputs = Fre_inputs.permute(0, 2, 1).unsqueeze(dim=1)
        fre_channel_att = self.F_channel_atten(Fre_inputs)
        Fre_inputs = rearrange(fre_channel_att, 'b o c f -> b o (c f)')
        Fre_inputs = torch.cat((Fre_inputs, Fre_inputs_tmp[:, 5, :20].unsqueeze(1)), dim=2)
        # b, t, c -> b, 1, c, t -> b t c
        Time_inputs = Time_inputs.permute(0, 2, 1).unsqueeze(dim=1)
        T_channel_att = self.T_channel_atten(Time_inputs)
        Time_inputs = T_channel_att.squeeze(dim=1).permute(0, 2, 1)
        # ================end channel attention
        # Fre
        # Fre_inputs = Fre_inputs.unsqueeze(2)
        Fre_inputs = rearrange(Fre_inputs, 'b c f  -> b f c')
        Fre_inputs = self.Fre_src_emb(Fre_inputs)
        # add class_token
        b, n, _ = Fre_inputs.shape
        cls_tokens = repeat(self.Fre_cls_token, '() n e -> b n e', b=b)
        Fre_inputs = torch.cat((cls_tokens, Fre_inputs), dim=1)

        # Position embedding
        Fre_inputs += self.Fre_pos_emb[:, :(n + 4)]
        Fre_pos_inputs = self.dropout(Fre_inputs)#+cls_tokens
        # Time
        b, n, _ = Time_inputs.shape
        Time_inputs = self.Time_src_emb(Time_inputs)
        cls_tokens = repeat(self.Time_cls_token, '() n e -> b n e', b=b)
        Time_inputs = torch.cat((cls_tokens, Time_inputs), dim=1)
        Time_inputs += self.Time_pos_emb[:, :(n+4)]
        Time_pos_inputs = self.dropout(Time_inputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        # enc_outputs = torch.cat((Fre_pos_inputs, Time_pos_inputs), dim=1)

        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(Fre_pos_inputs, Time_pos_inputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
