# 表示学习相关论文
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        # 将masking_ratio乘以num_patches得到要被mask的patch数量，使用int函数将结果取整并赋值给num_masked。
        num_masked = int(self.masking_ratio * num_patches)
        # 使用torch.rand函数在batch*num_patches的大小下生成随机数，使用topk函数找到随机数中最大的num_masked个数的索引值，返回结果中的indices属性是找到的索引。
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        # 使用torch.zeros函数创建一个大小为batch*num_patches的全零矩阵，然后使用scatter_函数将masked_indices对应的位置的值设为1，生成一个bool类型的掩码矩阵，表示哪些patch被mask了。
        # 其中scatter_()具体作用及使用方法可以参考官网https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html及知乎https://zhuanlan.zhihu.com/p/371889735
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        # torch.where()是PyTorch中的一个函数，用于根据给定的条件在两个张量之间进行选择元素。
        # torch.where(condition, x, y)
        # 其中，condition是一个布尔类型的张量，x和y是两个具有相同形状的张量。该函数返回一个新张量，其形状与x和y相同。
        # 对于每个元素，如果condition中的元素为True，则将x中的对应元素选择过来，否则将y中的对应元素选择过来。
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss
