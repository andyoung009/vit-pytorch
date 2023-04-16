import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# adaptive token sampling functions and classes

def log(t, eps = 1e-6):
    return torch.log(t + eps)

# 文章提到要对atention score进行求和形成一个概率分布，如果简单处理会因为分段函数出现不可导的情形，以此此处采用Gumbel分布确保概率分布可微可反向传播
# 这段代码实现了对 Gumbel 分布进行采样的功能，具体的作用解读如下：
# shape 参数表示采样出来的 Gumbel 分布张量的形状大小；
# device 参数表示采样出来的 Gumbel 分布张量的计算设备；
# dtype 参数表示采样出来的 Gumbel 分布张量的数据类型；
# eps 参数表示一个极小值，避免了分母为 0 的情况。
# 具体地，代码中首先利用 torch.empty 创建了一个形状为 shape 的未初始化张量 u，并在范围 [0, 1) 内进行了随机初始化；然后利用 Gumbel 分布的公式 Gumbel(x;μ,β)=−ln(−ln(u))
#  生成 Gumbel 分布采样结果，其中μ和β别为分布的位置和尺度参数，这里默认值为 0 和 1。
# 最终返回生成的 Gumbel 分布采样结果张量。Gumbel(x;μ,β)=−ln(−ln(u))生成 Gumbel 分布采样结果

# 顺便提一下文章DynamicViT文章也用到了这个方法，而ats把其一直作为了对比基线

# 参考https://zhuanlan.zhihu.com/p/166632315 辅助理解

# 此函数作用是定义sample_gumbel() 函数产生 Gumbel 噪声，即采样出服从Gumbel分布的噪声向量，形状与shape相同，用于离散空间采样向连续空间的映射
def sample_gumbel(shape, device, dtype, eps = 1e-6):
    u = torch.empty(shape, device = device, dtype = dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

# 该函数功能实现具体的细节没有理解！！
def batched_index_select(values, indices, dim = 1):
    # 获取values第2个维度之后的形状规格
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    # 这两行代码的作用是将indices的维度与values的维度对齐，使得在后续的操作中可以使用broadcasting。
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)

    # 在Python中，slice(None)等价于:，表示选取该维度上的所有元素。
    # 例如，对于一个二维数组x，x[:, 1]可以写成x[slice(None), 1]。
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    # 这行代码创建了一个 slice 对象，其起始位置是 1，结束位置是 dim + value_expand_len，步长默认为 1。
    # 这个 slice 对象可以用于索引操作，例如对列表、元组、数组等进行切片操作。在这个例子中，对于一个可切片的对象，这个 slice 对象可以截取索引为 1 到 5 的元素。
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    # 通过gather()函数来进行索引选择，其返回的结果张量形状为索引张量的形状与输入张量在指定维度上的形状拼接后的形状。
    return values.gather(dim, indices)

class AdaptiveTokenSampling(nn.Module):
    def __init__(self, output_num_tokens, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.device, attn.dtype

        # first get the attention values for CLS token to all other tokens

        cls_attn = attn[..., 0, 1:]

        # calculate the norms of the values, for weighting the scores, as described in the paper

        value_norms = value[..., 1:, :].norm(dim = -1)

        # weigh the attention scores by the norm of the values, sum across all heads

        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # normalize to 1

        normed_cls_attn = cls_attn / (cls_attn.sum(dim = -1, keepdim = True) + eps)

        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead

        pseudo_logits = log(normed_cls_attn)

        # mask out pseudo logits for gumbel-max sampling

        mask_without_cls = mask[:, 1:]
        # 这段代码定义了一个 mask_value 变量，用于在实现 self-attention 时对于 padding 位置的 token 进行 mask。
        # 具体来说，这里用到了 PyTorch 库中的 torch.finfo() 函数，用于返回浮点类型的数据类型的信息，包括它们的范围和精度等信息。
        # 这里调用 torch.finfo(attn.dtype) 返回 attn tensor 的数据类型的信息，再通过 .max 得到该数据类型的最大值。
        # 为了避免计算 softmax 时出现数值上的不稳定性，这里将 mask_value 初始化为该数据类型的最大值的一半（即将其除以 2），
        # 在实现 mask 操作时，将 padding 位置的 token 对应的位置的值设为该 mask_value，再将其输入 softmax 中，softmax 操作将会将该位置的概率值约束在很小的范围内。
        mask_value = -torch.finfo(attn.dtype).max / 2
        # 使用masked_fill函数将pseudo_logits张量中不需要的位置（即~mask_without_cls为True的位置）填充为指定的值mask_value。
        # 其中，mask_without_cls是一个布尔型张量，表示pseudo_logits中除了CLS位置之外的其他位置。
        # ~ 表示按位取反（bitwise NOT）运算符，对于二进制数中的每个位都进行翻转（0变为1，1变为0）
        # 为什么取反~？
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)

        # expand k times, k being the adaptive sampling number

        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k = output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device = device, dtype = dtype)

        # gumble-max and add one to reserve 0 for padding / mask

        sampled_token_ids = pseudo_logits.argmax(dim = -1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right

        # 那么可以使用 torch.unique() 获取该 sampled_token_ids 中的唯一值，即采样过一次就不用重复列出了：
        unique_sampled_token_ids_list = [torch.unique(t, sorted = True) for t in torch.unbind(sampled_token_ids)]
        # 将输入的张量序列上进行填充，使得每个序列的长度相等，第一个维度是batch，也就是unique_sampled_token_ids_list的个数
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first = True)

        # calculate the new mask, based on the padding

        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)

        new_mask = F.pad(new_mask, (1, 0), value = True)

        # prepend a 0 token id to keep the CLS attention scores

        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value = 0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h = heads)

        # gather the new attention scores

        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim = 2)

        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_num_tokens = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, *, mask):
        num_tokens = x.shape[1]

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 使用了张量 dots 的 masked_fill 方法，将 dots_mask 中对应为 False 的位置的值替换为 mask_value。
        # ~ 运算符是按位取反运算符，这里用于反转 dots_mask 中的布尔值。因此，masked_fill 方法将 dots_mask 中为 True 的位置的值替换为原始值
        # 将 dots_mask 中为 False 的位置的值替换为 mask_value

        # 为什么取反~，没有理解！！
        if exists(mask):
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        sampled_token_ids = None

        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask = mask)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), mask, sampled_token_ids

class Transformer(nn.Module):
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        assert len(max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse = True) == list(max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        self.layers = nn.ModuleList([])
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, output_num_tokens = output_num_tokens, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        b, n, device = *x.shape[:2], x.device

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = torch.ones((b, n), device = device, dtype = torch.bool)

        token_ids = torch.arange(n, device = device)
        token_ids = repeat(token_ids, 'n -> b n', b = b)

        for attn, ff in self.layers:
            attn_out, mask, sampled_token_ids = attn(x, mask = mask)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim = 1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim = 1)

            x = x + attn_out

            x = ff(x) + x

        return x, token_ids

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, max_tokens_per_depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, return_sampled_token_ids = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, token_ids = self.transformer(x)

        logits = self.mlp_head(x[:, 0])

        if return_sampled_token_ids:
            # remove CLS token and decrement by 1 to make -1 the padding
            # 因为0号token之前是cls token，现在需要除去
            token_ids = token_ids[:, 1:] - 1
            return logits, token_ids

        return logits
