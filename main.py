import copy
import math
import time

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        @param src: 原始数据[batch，length]
        @param tgt: 目标数据，shape同上
        @param src_mask: 布尔矩阵.[batch,1,length-1]
        @param tgt_mask:
        @return:
        """
        # 主要是一个解码操作，解码操作里面有一个编码的过程。毕竟编码码就是为了解码
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        "这里的encoder是一个模型 Encoder，src_embed也是个模型就是我们定义的Embeddings"
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    这是 decoder 后面跟着的 linear+softmax
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        "投影"
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=1)


def clones(module: nn.Module, N: int):
    """
    :param module:要被复制的网络
    :param N: 复制多少层
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    这是一个总的大编码器
    """

    def __init__(self, layer: nn.Module, N: int):
        """
        大编码器
        @param layer: EncoderLayer
        @param N: 默认为6
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    层归一化
    """

    def __init__(self, features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        # a_2 与 b_2 都是可训练的参数
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        :param x:形状为(batch.size, sequence.len, 512)
        :return:
        """
        # 这里的 keepdim 为true，会保持原有的形状，将这个维度设置为1.如果为false，那么就不存在这个维度了。
        # 因为所以的操作都是针对 512 维度的，所以都要对最后一个维度进行计算。
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 这里a_2 与b_2 论文中没有提到，貌似是为了多一些学习参数？
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    用来处理单个Sublayer的输出，该输出将继续被输入下一个Sublayer：
    执行的是 Add 和 Norm 的过程。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # 初始化 LayerNorm
        self.dropout = nn.Dropout(dropout)  # dropout 一种训练trick，随机屏蔽某些节点，来降低过拟合的风险

    def forward(self, x, sublayer):
        """
        :param x: （batches,sequence,512)
        :param sublayer:
        :return:
        """
        # 这个 sublayer是一个lambda函数，需要的是一个参数，
        # 这里是为了执行他们的forward方法
        # 下面的 x+ 就是一个残差相加过程。
        # 与论文的顺序不一样
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        这是一个单独的编码器，这里面包含了所有编码器的细节。
        :param size:
        :param self_attn:  多头注意力对象
        :param feed_forward: 前馈神经网络
        :param dropout:减少过拟合的
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 论文中，sublayer是将编码器看做了两部分，每一个部分就是一个sublaber。
        # 第一个sublayer包含了多头注意力和一个add&Norm，第二个sublayer包含了前馈神经网络和一个 add&Norm
        # 但是这里是一个SublayerConnection，是一个Connection,只是一个连接过程。
        self.sublayer = clones(SublayerConnection(size=size, dropout=dropout), N=2)
        self.size = size

    def forward(self, x, mask):
        # 执行的是 SublayerConnection 的 forward 方法
        # 首先使用多头注意力运行一波，然后，运行 SublayerConnection 的forward 方法
        # 这里使用的是 lambda 方法。 这个参数是y，对应SublayerConnection的参数是sublayer。
        # 是 sublayer 中的参数 self.norm(x) 看做y。
        # self.self_attn(self.norm(y), self.norm(y), self.norm(y), mask)

        # lambda 函数是匿名函数， labmda 参数写在冒号前面， lambda x,y：x*y ，就是参数是x,y，执行的内容是x*y
        # 第一个子层链接，是针对多头注意力网络的，x+MultiAttentionHead,多头输入的是k,q,v 以及mask【看起来是srcmask】
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """

        @param x: tgt 目标数据经过Embedding的输出【batch*seq_length-1*512】
        @param memory: 编码器的输出
        @param src_mask: 应该是一个全为true的布尔矩阵
        @param tgt_mask:一个下三角矩阵[batch* seq_length-1*seq_length-1]
        @return:
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    单个的解码层
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: tgt 目标数据（label）
        :param memory: 来自源语言序列的Encoder之后的输出，作为memory
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        m = memory
        # Masked Multi-Head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Multi-Head Attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)  # Feed Forward


def attention(query, key: torch.Tensor, value: torch.Tensor, mask=None, dropout=None):
    """
    注意力机制
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:Attention(K,Q,V) ,softmax(K*Qt)
    """
    # 这是维度，query.size()默认返回 torch.Tensor ，带上形参数的名为 dim ，返回为 int
    d_k = query.size(-1)  # 64
    # 论文中的计算k,q,v的公式,scores的shape是[nbatches,head,seq_length,seq_length]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 在编码器中，mask都是1，也就是说下面scores不会发生变化
        # masked_fill 第一个参数必须为 布尔矩阵，后面是要被填充的。
        # 在python 中， True 为非零，False 为 0

        # 这是一个掩码操作，目的将mask为false的的都设置为-1e9。
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力部分，这里的多头，是将原始的切割为了多个部分。
    """

    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: 头数
        :param d_model: 输入的维度，也就是将一个字用多少维度的向量来表示，一般是 512
        :param dropout:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # //运算符表示向下取整除，它会返回整除结果的整数部分,这是每个头掌管多少维度。 512/8=64
        self.h = h
        # 输入的维度是 d_model论文中是512 ,输出的维度也是 d_model,这里复制了4份，其中前3份是为k,q,v用的，最后一个是为多头最终的线性输出准备的。
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 这个似乎没什么用
        self.dropout = nn.Dropout(p=dropout)  # nn.Dropout 继承自 _DropoutNd ，而 _DropoutNd 使用了 p 来进行初始化

    def forward(self, query, key, value, mask: torch.Tensor = None):
        """
        @param query: shape为[nbatches,seq_length,d_model]
        @param key:
        @param value:
        @param mask:
        @return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
            # mask 是一个batch *1 *seq_length 的掩码矩阵，这里unsqueeze将他变成[batch*1*1*seq_length]的矩阵，与query，key等的维度相同，可以通过广播体质扩充进而使用mask_fill方法
        nbatches = query.size(0)
        # 这里的for 循环一共执行了3次。这里 self.linears 有4层，而（query,key,value）这个列表有3个。所以这个for循环执行了3次，以此给了k，q，v
        # 每一个线性层用x作为参数执行 Linear 的forward方法，返回Tensor，然后view强制reshape。
        # zip的用法非常巧妙，它是逐个进行的，
        # self.linears[0](query)
        # self.linears[1](key)
        # self.linears[2](value)
        # l(x)的size是（nbatch*d_model）
        # 转型为（n_batch,seq_length,头数，每个头的维度【d_model//头数】）
        # 转置为(n_batch,头数,seq_length，64）
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, self.dropout)
        # contiguous 是使其连续的意思，因为转置后，底层的存储不再连续。只有连续的，才能使用view方法
        # x的shape是 [nbatches,head,seq_length,每个头的维度64]
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))
        # x转置后为[nbatches,seq_length,head,每个头的维度64],结果view操作后变为[nbatches,seq_length,512]
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    这个是编码器，解码器中的 FeedForward 层
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 输入张量的维度
        :param d_ff: 隐藏层的尺寸
        :param dropout: 开启dropout丢弃的概率
        :return:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 512投影到2048
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # relu 激活函数。self.w_1(x).relu()是论文中 max(0,xW1+b1) 的实现
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """

        :param d_model: 嵌入向量的维度，也就是使用多少维来表示一个词。论文中使用512
        :param vocab: 词典的大小，如果总共有5000个词，那么输入就是5000。也就是词表的大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  # 512

    def forward(self, x):
        # 不是很清楚这个是
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码输入
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (5000,512)
        position = torch.arange(0, max_len).unsqueeze(1)  # 5000->(5000,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0)  # 为了给 batch 预留位置
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 的样式为 （batch ，sequence，d_model）
        :param x:
        :return:
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1) -> EncoderDecoder:
    """
    汇总 TransFormer 模型
    :param src_vocab:
    :param tgt_vocab:
    :param N:解码器，编码器堆了几层
    :param d_model:输入的维度，也就是将一个字用多少维度的向量来表示，一般是 512
    :param d_ff:这里在每个解码器/编码器中的全连接层用到，512->2045->512
    :param h:一共有几个头，默认为8
    :param dropout:随机遮盖多少神经元
    :return:
    """
    c = copy.deepcopy  # typedef 一个函数
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)  # 每个解码/编码器中的 全连接层
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout), N),
        decoder=Decoder(
            DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=dropout), N),
        # 这里的 nn.sSqquential 是指一个序列，这段代码中，序列加入了两个网络，一个是 Embeddings，另一个是位置编码网络
        src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=src_vocab),
                                c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=tgt_vocab),
                                c(position)),
        generator=Generator(d_model=d_model, vocab=tgt_vocab)
    )
    # 随机初始化参数，非常重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    def __init__(self, src, tgt=None, pad=2):
        """
        :param src: 输入序列 （batch,length_src)
        :param tgt: 输出序列 (batch,length_tgt)
        :param pad: <bank> 符号，这里虽然是2，但是运行中调用的是0
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # src_mask 是一个 (batch,1,length_src) 的布尔矩阵
        if tgt is not None:
            """
            在机器翻译的任务中，通常使用“teacher forcing”技术来训练模型。
            这意味着在训练期间，我们将目标序列的每个位置的正确单词提供给模型，以便其预测下一个单词。
            为了在训练期间将正确的单词提供给模型，我们需要将目标序列向右移动一个位置，即用第二个到最后一个单词作为输入，并将第一个到倒数第二个单词作为目标输出。
            这就是代码中的 self.tgt = tgt[:, :-1] 所做的事情。
            因此，在初始化Batch对象时，self.tgt 中的最后一个单词被删除，因为它将成为模型在训练期间的输出，而不是作为模型的输入提供给模型。
            """
            self.tgt = tgt[:, :-1]  # 舍弃最后一个字符，维度 （batch,length_tgt -1)
            self.tgt_y = tgt[:, 1:]  # 舍弃第一个字符，这个是干啥的，貌似在计算损失的时候用，不太清楚。
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        :param tgt: 输出序列，维度 （batch,length_tgt-1）
        :param pad: 0
        :return:
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)  # tgt_mask 已经是一个布尔矩阵了。 维度（batch,1，length_tgt -1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # subsequent_mask 生成了一个 （1，length_tgt -1，length_tgt -1）的下三角矩阵。相 & 的结果就是（batch，length_tgt -1，length_tgt -1）维度的布尔矩阵
        # tgt_mask 是个batch的seq_length-1*seq_length-1的下阶梯矩阵
        return tgt_mask


def subsequent_mask(size: int) -> torch.Tensor:
    """
    生成一个下三角矩阵，只有decoder使用
    :param size:
    :return: 下三角布尔矩阵，【1，size，size】
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)  # 上三角矩阵，同时，对角线上移1位
    return subsequent_mask == 0  # 巧妙地转成了下三角矩阵


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,
        model: EncoderDecoder,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # batch.src_mask 是一个全为true的矩阵，【80*1*10】，tgt_mask是一个[80*9*9]的下三角true矩阵
        # 这里的forward执行的是 EncoderDecoder 的forward
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]]).cuda()
    return crit(predict.log(), torch.LongTensor([1])).data


def data_gen(V, batch_size, nbatches):
    """
    这是一个数据产生器
    @param V: 限定范围
    @param batch_size: 一个批次的大小
    @param nbatches: 要产生多少轮数
    """
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10)).cuda()  # 设置数据范围
        data[:, 0] = 1  # 第一个数据一定是1，但是不知道为什么这么设计
        """
        在机器翻译任务中，通常会在源序列的开头添加一个特殊的起始标记，如 <s>。
        同样地，在语言模型任务中，通常会在输入序列的开头添加一个类似的标记，以指示模型开始生成文本。
        这个标记通常是一个特殊的单词，例如 <s> 或 <start>。
在这里，将数据范围设置为 (batch_size, 10) 并将第一个数据设置为1是为了在数据中添加一个起始标记。
这是一个常见的设计，它可以帮助模型更好地学习生成文本或者在生成文本时更好地处理边界情况。
当模型遇到这个起始标记时，它会知道它需要从这里开始生成文本，而不是从输入序列的开头开始。
这个起始标记不是真正的单词，而只是一个用于协助模型的特殊标记，通常在训练数据中是不存在的。
        """
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).cuda()
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def example_simple_model():
    V = 10
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).cuda()
    model = make_model(src_vocab=V, tgt_vocab=V, N=6).cuda()  # 单词表
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 2
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).cuda()
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).cuda()
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == '__main__':
    example_simple_model()
