from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        # 该位置嵌入选取的参数是根据输入的x的具体shape确定的，因为下面的数组的角标做了切片，一共选取了前（h*w）个参数
        # 因此设计pos_emb时只需要保证在第一层时有足够的位置嵌入参数
        # 后续则由于聚合函数的作用，block的数量一直降低，因此参数一定是够的！
        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class NesT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size        # 图像尺寸上有多少patch_size
        # 论文图1中num_hierarchies=3
        blocks = 2 ** (num_hierarchies - 1)     #   = 4（block在特征图维度上包含patch_size的个数）

        # 特征图中包含多少个blocks呢？ 
        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across hierarchy

        # num_hierarchies 是金字塔中总共的层数，此处为3。
        # hierarchies 是一个倒序排列的整数列表，表示金字塔中每一层的编号，例如 [3, 2, 1, 0] 表示第一层编号为 3，第二层编号为 2，以此类推。
        # mults 是一个倒序排列的整数列表，表示每一层的空间尺度相对于最底层的空间尺度的比例因子，例如 [8, 4, 2, 1] 表示最底层的空间尺度是 8 像素，第二层是 4 像素，以此类推。
        # layer_heads 是一个倒序排列的整数列表，表示每一层的头数（即注意力机制中的头数），例如 [32, 16, 8, 4] 表示最顶层的头数为 32，第二层是 16，以此类推。
        # layer_dims 是一个倒序排列的整数列表，表示每一层的特征维度数，例如 [256, 128, 64, 32] 表示最顶层的特征维度数为 256，第二层是 128，以此类推。
        hierarchies = list(reversed(range(num_hierarchies)))    # [2,1,0]
        mults = [2 ** i for i in reversed(hierarchies)]         # [1,2,4]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            LayerNorm(patch_dim),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
            LayerNorm(layer_dims[0])
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape

        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            # 此处是不是维度信息有问题？是不是应该按照Line136中block的数量将输入的patch再进行尺寸上的划分呢？
            # 现在第一循环block_size=1，则h和w表示的仍然是特征图在宽高方向上有多少个patches，而不是多少个blocks
            # 那么必然导致在位置编码加法运算时出现问题！和line136冲突了！
            # 应修改
            # block_size = 2 ** (level + num_hierarchies - 1) 
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        return self.mlp_head(x)
