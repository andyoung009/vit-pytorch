import copy
import random
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

# 这是一个装饰器函数，它接受一个字符串参数cache_key，返回一个函数inner_fn作为装饰器。
# inner_fn接受一个函数参数fn，并返回一个新的函数wrapper作为装饰器的结果。
# wrapper函数将原函数fn进行了封装，并实现了单例模式，每次调用时都返回同一个对象实例。
# 具体包装器函数检查修饰类的实例是否已经存在于具有给定cache_key的缓存中。如果存在，则返回缓存的实例。
# 否则，它通过使用提供的参数调用原始函数fn创建一个新实例，使用给定的cache_key缓存该实例，并返回该实例。
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# 这个函数接受一个PyTorch模型module作为参数，返回该模型的第一个参数所在的设备。
# 具体来说，它通过module.parameters()获取模型的参数列表，然后调用next()获取第一个参数，并从该参数的device属性中获取设备信息。
def get_module_device(module):
    return next(module.parameters()).device

# 函数接受一个PyTorch模型model和一个布尔值参数val作为参数，将模型中所有的参数的requires_grad属性设置为val。
# 函数通常用于模型的参数冻结或解冻，当val为True时，所有参数都会被设置为需要梯度计算，当val为False时，所有参数都会被设置为不需要梯度计算。
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss function # (algorithm 1 in the paper)

# 代码定义了一个损失函数loss_fn，它接受以下参数：

# teacher_logits：老师网络的输出 logits，形状为 [batch_size, num_classes]。
# student_logits：学生网络的输出 logits，形状为 [batch_size, num_classes]。
# teacher_temp：老师网络的温度参数。
# student_temp：学生网络的温度参数。
# centers：形状为 [num_classes, num_features] 的矩阵，用于计算老师网络的 softmax 函数的分母。
# eps：一个小的常数，用于数值稳定性，避免取对数时出现 NaN。
# 该损失函数的实现逻辑如下：

# 将老师网络的输出 logits teacher_logits 从计算图中分离出来，以避免反向传播时对老师网络产生影响。
# 将学生网络输出 logits student_logits 除以学生温度参数 student_temp，并对结果进行 softmax 计算，得到学生网络的预测概率分布 student_probs。
# 将老师网络的输出 logits teacher_logits 减去矩阵 centers，然后除以老师温度参数 teacher_temp，并对结果进行 softmax 计算，得到老师网络的预测概率分布 teacher_probs。
# 计算交叉熵损失，即将 teacher_probs 与 student_probs 的对数相乘，然后取负数，最后在最后一个维度上求和并求平均值。
# 该损失函数的实现逻辑类似于知识蒸馏（knowledge distillation）算法中的损失函数，用于将老师网络的知识传递给学生网络。
# 具体来说，它通过将老师网络的输出 logits 作为“软目标”（soft target）来帮助学生网络学习更加鲁棒的决策边界。
# 因此，该损失函数通常用于训练一个更小的学生网络，以达到与一个大的老师网络相似的性能

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    # 见论文第3页式2
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# augmentation utils

# 定义了一个名为RandomApply的PyTorch模块，用于在给定概率下随机应用给定的函数。
# 在代码中，模块包含两个参数：
# fn: 要应用的函数。
# p: 应用函数的概率。
# 在模块的前向方法forward中，随机生成一个0到1之间的随机数，如果随机数大于p，则直接返回输入x，否则将输入x作为参数传递给函数fn，并返回fn的输出。
# 这个模块可以用来实现一些随机数据增强技术，例如随机旋转、裁剪、反转等。在训练深度学习模型时，将这个模块应用到输入数据上可以增加模型的鲁棒性和泛化能力。
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        # 对于给定的输入x，x.norm(dim = 1, keepdim = True)计算了每个样本的L2范数，即每行向量的模长。然后使用clamp方法将范数限制在eps和正无穷之间，避免出现除数为0的情况
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        # 行向量维度上归一化
        return x / norm

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        # 维度信息的元组
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        # zip打包成对的输入维度和输出维度信息，生成不同的线性层激活层连接起来
        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),   #归一化
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False


    # 这段代码定义了一个名为_find_layer的方法，用于从神经网络中查找指定的层。
    # 该方法接收一个self对象作为输入，其中包含三个属性：
    # self.layer: 指定要查找的层，它可以是一个字符串，表示层的名称；也可以是一个整数，表示层在网络中的索引；如果不是这两种类型，则返回None。
    # self.net: 神经网络，它是一个PyTorch模型对象。
    # 该方法首先判断self.layer的类型，如果是字符串，则通过named_modules方法获取网络中所有的模块，使用get方法根据层名称获取指定层的实例；
    # 如果是整数，则通过children方法获取网络中所有的子模块，根据索引找到指定层的实例；如果不是这两种类型，则返回None。
    # 最终，该方法返回查找到的层的实例。这个方法可以用来在神经网络中查找指定的层，例如从预训练模型中提取某个层的特征表示，或者在训练过程中对某个层的参数进行优化等
    # 关于_find_layer相关内容结合印象笔记学习记录第457条理解
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    # 钩子函数，用于获取网络中的中间层输出。具体来说，该函数将被注册到模型的某个中间层上，每次该中间层被调用时
    # 该函数都会被自动调用，并在该层的输入和输出上执行一些操作。
    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.flatten(1)

    # 注册钩子函数
    def _register_hook(self):
        layer = self._find_layer()
        # 层非空，则注册该层的钩子函数
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    # _get_projector() 方法：这是一个带有装饰器 @singleton('projector') 的私有方法。它接受一个名为 hidden 的张量，并使用其形状来
    # 创建一个多层感知器（MLP）对象 projector，并将其转移到 hidden 所在的设备上。最后返回 projector 对象。
    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    # get_embedding() 方法：该方法接受一个名为 x 的张量，并返回该张量在经过一个预先定义的神经网络模型后的输出。如果在初始化时指定了 layer 属性，则返回该层的输出。
    # 如果没有指定，则返回最终输出。如果 hook_registered 属性为 False，则会注册一个钩子函数 _register_1hook()，该函数将在前向传递过程中记录隐藏层的输出。
    # 最后返回隐藏层的输出张量 hidden。
    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_1hook()

        # 清空 self.hidden 字典。
        self.hidden.clear()
        # 调用神经网络模型 self.net 并将输入张量 x 作为参数传递给它，运行前向传递过程；
        # 在前向传递过程中，已注册的钩子函数 _register_1hook() 将在指定的隐藏层处记录输出值。
        _ = self.net(x)
        # 从 self.hidden 字典中获取设备为 x.device 的隐藏层输出张量 hidden。
        hidden = self.hidden[x.device]
        # 最后，再次清空 self.hidden 字典以避免钩子函数记录不需要的输出张量。
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    # forward() 方法：该方法接受一个名为 x 的张量和一个名为 return_projection 的布尔值，表示是否返回投影后的张量。
    # 首先调用 get_embedding() 方法获取输入张量的隐藏层输出 embed。
    # 如果 return_projection 为 False，则直接返回 embed。否则，调用 _get_projector() 方法创建一个 MLP 对象 projector
    # 并将 embed 作为输入，返回 projector(embed) 投影后的张量和 embed 原始的张量。
    def forward(self, x, return_projection = True):
        embed = self.get_embedding(x)
        if not return_projection:
            return embed

        projector = self._get_projector(embed)
        return projector(embed), embed

# main class

class Dino(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None
    ):
        super().__init__()
        self.net = net

        # default BYOL augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # local and global crops

        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        # teacher上应用stop-gradient (sg)操作符，只通过student来传播梯度。教师参数使用学生参数的指数移动平均(ema)进行更新
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)

        loss_fn_ = partial(
            loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        # 这是一个计算损失函数的代码片段，其中包含了两个投影向量 teacher_proj_one 和 teacher_proj_two，以及两个投影向量 student_proj_one 和 student_proj_two。
        # 这些向量在进行对比学习时通常用于比较教师模型和学生模型的相似性。
        # 该代码片段使用了两个损失函数 loss_fn_() 的平均值来计算总损失。损失函数 loss_fn_() 可以是任何可微分函数，通常用于测量两个向量之间的距离或相似性。
        # 在这个例子中，计算了 teacher_proj_one 和 student_proj_two 之间的损失，以及 teacher_proj_two 和 student_proj_one 之间的损失，然后将它们的平均值作为总损失返回。
        # 这种方法的目的是鼓励教师模型和学生模型在两个方向上保持相似性，从而提高模型的鲁棒性和泛化能力。
        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss
