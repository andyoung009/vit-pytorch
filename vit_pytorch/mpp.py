# 表示学习相关论文，但是vit-pytorch中没有给出具体论文链接，只给了Masked Patch Prediction的代码和作者GitHub主页
import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

# 按照张量t的前2个维度生成一个掩码，掩码生成的规则是从（0，1）的均匀分布与预设概率值prob的大小来判断，返回的掩码矩阵是二值化的bool型张量
def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob

# 此处的prob为掩码patches个数占总patches的比率
def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    # 使用torch.rand函数在batch*seq_len的大小下生成随机数，使用topk函数找到随机数中最大的max_masked个数的索引值，返回结果中的sampled_indices属性是找到的索引。
    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    # 使用torch.zeros函数创建一个大小为batch*num_patches的全零矩阵，然后使用scatter_函数将masked_indices对应的位置的值设为1
    # 生成一个bool类型的掩码矩阵，表示哪些patch被mask了。
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


# mpp loss


class MPPLoss(nn.Module):
    def __init__(
        self,
        patch_size,
        channels,
        output_channel_bits,
        max_pixel_val,
        mean,
        std
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        # 转换维度
        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device

        # 这段代码是计算分桶的桶大小（bin size）的公式。其中，mpv是目标数据的最大值（most positive value），bits是将目标数据量化所使用的比特数（bits per channel）。具体解释如下：
        # 分桶的目的是将一定范围内的连续数值划分为若干个离散的区间，每个区间就称为一个桶。对于目标数据，将其按照一定的规则（例如等间距或等频率等）划分成多个桶，有助于进行数据分析、特征提取等操作。
        # mpv是目标数据中的最大值，根据其与最小值之间的范围可以确定数据的取值范围。
        # bits是将目标数据量化所使用的比特数，也就是数据压缩的程度。比特数越少，所能表示的数值范围就越小。
        # 该公式将mpv除以2的bits次方，得到了每个桶的大小（bin size）。因为每个桶的宽度相等，所以通过该公式可以动态计算出数据的分布情况并进行离散化处理。
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = target.clamp(max = mpv) # clamp just in case
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1 = p, p2 = p).contiguous()

        # 这行代码的作用是在设备 device 上生成一个张量 channel_bins，其中包含了从 bin_size 开始、以 bin_size 为步长、直到 mpv 但不包括 mpv 的一系列数值。
        # 具体来说，这个张量包含了 mpv / bin_size - 1 个数值。例如，如果 mpv=3，bits=8，那么 bin_size 的计算结果为 3 / 256 = 0.01171875，
        # channel_bins 的计算结果为 tensor([0.0117, 0.0234, 0.0352, 0.0469, ..., 2.9258, 2.9375])。
        channel_bins = torch.arange(bin_size, mpv, bin_size, device = device)
        # 这行代码的作用是将 avg_target 按照给定的 channel_bins 进行分桶，并返回每个值所属的桶的索引。具体来说，channel_bins 是一个一维的 Tensor，表示每个桶的右端点。
        # 例如，如果 channel_bins = [0.5, 1.5, 2.5]，则表示将数值分为四个桶：(-inf, 0.5]、(0.5, 1.5]、(1.5, 2.5]、(2.5, +inf)；
        # avg_target 是一个 Tensor，其形状为 (batch_size, num_channels, height, width)，其中 batch_size 表示批次大小，
        # num_channels 表示通道数，height 和 width 分别表示图像的高和宽。avg_target 中的每个元素表示输入图像中的一个像素在各个通道上的均值；
        # torch.bucketize 函数将 avg_target 按照 channel_bins 进行分桶，并返回每个值所属的桶的索引。返回的 discretized_target 是一个整数类型的 Tensor，其形状与 avg_target 相同，
        # 表示每个像素在各个通道上所属的桶的索引。例如，如果某个像素在第 0 个通道上的均值为 1.2，而 channel_bins 为 [0.5, 1.5, 2.5]，则该像素在第 0 个通道上的索引为 1，表示它所属于 (0.5, 1.5] 这个桶。
        discretized_target = torch.bucketize(avg_target, channel_bins)

        # 这段代码的功能是将每个通道的均值映射到指定的离散区间上，然后通过对应的二进制掩码将它们转换为整数标签。具体来说，它首先将均值张量avg_target与通道区间bin_size进行分桶操作，并将结果保存在名为discretized_target的张量中。
        # 然后，它通过创建一个bin_mask张量，该张量是形状为(1, 1, c)的三维张量，其中c是通道数，bin_mask的每个元素都是2的bits次方的幂次方，bits是给定的输出通道位数。接下来，代码使用重组操作将bin_mask的形状更改为(1, 1, c)。
        # 最后，它将离散化的目标标签与bin_mask张量相乘，并使用sum函数将结果加和在通道维度上，从而将每个通道的离散化标签转换为一个整数标签，并将其存储在target_label中。
        bin_mask = (2 ** bits) ** torch.arange(0, c, device = device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = torch.sum(bin_mask * discretized_target, dim = -1)

        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


# main class


class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        output_channel_bits=3,
        channels=3,
        max_pixel_val=1.0,
        mask_prob=0.15,
        replace_prob=0.5,
        random_patch_prob=0.5,
        mean=None,
        std=None
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std)

        # output transformation
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p,
                          p2=p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = transformer.to_patch_embedding[-1](masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss
