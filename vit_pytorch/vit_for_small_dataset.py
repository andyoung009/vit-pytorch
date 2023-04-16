# https://github.com/aanna0701/SPT_LSA_ViT
# https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/SPT.py
# 官方代码库见上
from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        # from the cursor 
        # The line of code in the LSA class constructor initializes the temperature attribute as a learnable parameter. 
        # The value of this parameter is set to the logarithm of dim_head raised to the power of negative 0.5. 
        # This value is used to scale the dot product between the query and key vectors in the attention mechanism, as seen in the forward method of the LSA class.
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        # 创建attention score的主对角线的掩码
        # 这段代码的作用是创建一个对角线为True，其余元素为False的布尔类型的掩码（mask），并将其应用于给定的张量dots。
        # 掩码中的True值会将dots对应位置的值置为给定数据类型（dtype）的最小值的相反数。
        # 该掩码的作用是在计算self-attention时，避免计算每个位置与自身的点积，因为这些点积不会提供有用的信息，因为它们只表示相同位置的向量。
        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/SPT.py

# import torch
# from torch import nn
# from einops import rearrange
# from einops.layers.torch import Rearrange
# import math

# class ShiftedPatchTokenization(nn.Module):
#     def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False):
#         super().__init__()
        
#         self.exist_class_t = exist_class_t
        
#         self.patch_shifting = PatchShifting(merging_size)
        
#         patch_dim = (in_dim*5) * (merging_size**2) 
#         if exist_class_t:
#             self.class_linear = nn.Linear(in_dim, dim)

#         self.is_pe = is_pe
        
#         self.merging = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim)
#         )

#     def forward(self, x):
        
#         if self.exist_class_t:
#             visual_tokens, class_token = x[:, 1:], x[:, (0,)]
#             reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#             out_visual = self.patch_shifting(reshaped)
#             out_visual = self.merging(out_visual)
#             out_class = self.class_linear(class_token)
#             out = torch.cat([out_class, out_visual], dim=1)
        
#         else:
#             out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#             out = self.patch_shifting(out)
#             out = self.merging(out)    
        
#         return out
        
# class PatchShifting(nn.Module):
#     def __init__(self, patch_size):
#         super().__init__()
#         self.shift = int(patch_size * (1/2))
        
#     def forward(self, x):
     
#         x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
#         # if self.is_mean:
#         #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
#         以下代码的作用是对输入的特征图进行扩充，以便提取更多的上下文信息。具体来说，代码的执行过程如下：
# 使用torch.nn.functional.pad()函数对输入特征图x进行零填充，填充的大小为(self.shift, self.shift, self.shift, self.shift)。这样做是为了防止在后续的操作中越界。
# 将填充后的特征图x_pad分别沿着左、右、上、下四个方向移动2个位置，得到四个新的特征图x_l2、x_r2、x_t2和x_b2。这些特征图对应了原始特征图中每个像素点周围2个像素点的信息。
# 将原始特征图x和四个移动后的特征图x_l2、x_r2、x_t2和x_b2沿着通道维度进行拼接，得到一个新的特征图x_cat，该特征图包含了更多的上下文信息。返回拼接后的特征图x_cat。
#         """ 4 cardinal directions """
#         #############################
#         这行代码中的切片是对填充后的特征图进行的，具体解释如下：
            # : 表示对所有样本进行操作；
            # : 表示对所有通道进行操作；
            # self.shift:-self.shift 表示对从第self.shift个位置到倒数第self.shift个位置的行/列进行操作；
            # :-self.shift*2 表示对从开头到倒数第self.shift*2个位置的列进行操作。
            # 因此，x_l2表示的是在填充后的特征图x_pad中，去掉上下左右各self.shift个像素点后剩余的部分，但是只保留左边的第self.shift*2列。也就是说，x_l2包含了原始特征图中每个像素点左边2个像素点的信息，但是没有包含右边和上下方向的信息。
#         # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
#         # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
#         #############################
        
#         """ 4 diagonal directions """
#         # #############################
#         x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
#         x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
#         x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
#         x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
#         x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
#         # #############################
        
#         """ 8 cardinal directions """
#         #############################
#         # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
#         # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
#         # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
#         # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
#         # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
#         # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
#         #############################
        
#         # out = self.out(x_cat)
#         out = x_cat
        
#         return 

# 关于具体shift Patch Tokenization的模式可以参考论文的最后一部分附录和上述代码
class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        # *5是因为前向函数中分布把四个对角线方向和原来的x，5倍于原来的x的通道
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        # 代码段的作用是对给定的输入张量x进行四个方向的移位操作，并将这些移位后的张量存储在列表shifted_x中。
        # 移位的方向由shifts元组中的四个元组指定，每个元组对应一种方向，具体来说，第一个元组表示向左移动一个位置，第二个元组表示向右移动一个位置
        # 第三个元组表示向上移动一个位置，第四个元组表示向下移动一个位置。(与文章3.1中的有区别)每个移位操作通过在x的边界处进行填充（padding）来实现，填充值默认为0。
        # 需要指出的是，这种移位操作并没有扩大x的感受野，因为它们仅仅是将x中的信息在空间维度上进行了重排，可以参考文中图2的系统架构图进行理解。
        """"
        如果要扩大感受野,应该借鉴Swin-Transformer的滑窗的方法,先把x利用torch.roll()函数实现滚动，然后再取出相应分割的patches
                # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
                ……
        """
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        # 沿着1维度拼接，所以原来的通道信息变化'c -> 5c'
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    # 在这个函数定义中，表示使用关键字参数传递参数。这意味着该函数可以通过名称传递任意数量的参数，而无需将其全部列出。
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # 此处按照不同的pool type执行相应操作
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
