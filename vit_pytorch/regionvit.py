import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def divisible_by(val, d):
    return (val % d) == 0

# helper classes

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# transformer classes

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rel_pos_bias = None):
        h = self.heads

        # prenorm

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add relative positional bias for local tokens

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class R2LTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        window_size,
        depth = 4,
        heads = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.window_size = window_size
        rel_positions = 2 * window_size - 1
        # Swin-Transformer的相对位置定义采用的是torch.Parameter()，但其实生成的相对位置参数表的规格是一样大小的，异曲同工
        self.local_rel_pos_bias = nn.Embedding(rel_positions ** 2, heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))

    def forward(self, local_tokens, region_tokens):
        # 获取设备
        device = local_tokens.device
        # 函数首先获取输入张量的维度信息，然后将local_tokens和region_tokens重排为二维张量，维度顺序为(batch_size, h*w, channels)，其中h和w是输入张量的高度和宽度
        lh, lw = local_tokens.shape[-2:]
        rh, rw = region_tokens.shape[-2:]
        window_size_h, window_size_w = lh // rh, lw // rw

        local_tokens = rearrange(local_tokens, 'b c h w -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b c h w -> b (h w) c')

        # calculate local relative positional bias
        # 经过对比，发现此次的代码和Swin-Transformer的代码真的是一个思路！第一次看到Swin-transformer时不理解，现在明白了（欧几里得方法实现相对位置编码）
        
        h_range = torch.arange(window_size_h, device = device)
        w_range = torch.arange(window_size_w, device = device)

        grid_x, grid_y = torch.meshgrid(h_range, w_range, indexing = 'ij')
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (self.window_size - 1)
        bias_indices = (grid * torch.tensor([1, self.window_size * 2 - 1], device = device)[:, None, None]).sum(dim = 0)
        rel_pos_bias = self.local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        # 因为下文将local embedding和region embedding合并了(torch.cat())，因此需要补足相对位置偏置矩阵
        rel_pos_bias = F.pad(rel_pos_bias, (1, 0, 1, 0), value = 0)

        # go through r2l transformer layers

        for attn, ff in self.layers:
            region_tokens = attn(region_tokens) + region_tokens

            # concat region tokens to local tokens

            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h = lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1 = window_size_h, p2 = window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')

            # do self attention on local tokens, along with its regional token

            region_and_local_tokens = torch.cat((region_tokens, local_tokens), dim = 1)
            # 拼接后不影响具体注意力的计算，因为attention计算过程中n的维度始终没有参与Linear()的运算，见attention类定义
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias = rel_pos_bias) + region_and_local_tokens

            # feedforward

            region_and_local_tokens = ff(region_and_local_tokens) + region_and_local_tokens

            # split back local and regional tokens

            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h = lh // window_size_h, w = lw // window_size_w, p1 = window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n = rh * rw)

        local_tokens = rearrange(local_tokens, 'b (h w) c -> b c h w', h = lh, w = lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b c h w', h = rh, w = rw)
        return local_tokens, region_tokens

# classes

class RegionViT(nn.Module):
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        window_size = 7,
        num_classes = 1000,
        tokenize_local_3_conv = False,
        local_patch_size = 4,
        use_peg = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3,
    ):
        super().__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'

        self.local_patch_size = local_patch_size

        region_patch_size = local_patch_size * window_size
        self.region_patch_size = local_patch_size * window_size

        init_dim, *_, last_dim = dim

        # local and region encoders

        if tokenize_local_3_conv:
            self.local_encoder = nn.Sequential(
                nn.Conv2d(3, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            )
        else:
            self.local_encoder = nn.Conv2d(3, init_dim, 8, 4, 3)

        self.region_encoder = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = region_patch_size, p2 = region_patch_size),
            nn.Conv2d((region_patch_size ** 2) * channels, init_dim, 1)
        )

        # layers

        current_dim = init_dim
        self.layers = nn.ModuleList([])

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and use_peg

            self.layers.append(nn.ModuleList([
                Downsample(current_dim, dim) if need_downsample else nn.Identity(),
                PEG(dim) if need_peg else nn.Identity(),
                R2LTransformer(dim, depth = num_layers, window_size = window_size, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

            current_dim = dim

        # final logits

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        *_, h, w = x.shape
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'

        # 相对于local_tokens，region_tokens的特征量级小很多，按照输入img=(b, 3, 224, 224)计算，在R2LTransformer类前向函数中tokens的处理工程中
        # local_tokens，region_tokens的embedddings在个数维度比为49：1(b*8*8,7*7,dim):(b*8*8,1,dim)，实际上全局特征大大弱化了,具体参考R2LTransformer类前向函数
        local_tokens = self.local_encoder(x)
        region_tokens = self.region_encoder(x)

        for down, peg, transformer in self.layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)
        
        # 此处直接返回region_tokens的平均池化，感觉结合region_tokens运算时的维度信息（相对于local_tokens为1维度），它似乎起到了cls_token的分类作用
        return self.to_logits(region_tokens)
