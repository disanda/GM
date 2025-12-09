import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 设置使用的 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = './Glow_v6_hidden=256_bflow=3-30-[-0.5,+0.5]_fmnist'
os.makedirs(folder_path, exist_ok=True)

# 超参数设置
n_blocks = 3        # 多尺度块数量
n_flows = 30        # 每个块的 flow 数量
hidden_channels = 256
batch_size = 128
epochs = 100
image_size = 32
in_channel = 1
lr = 3e-4

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),])  # 数据范围为 [0,1]

#dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#####################################
# 工具函数
#####################################

def add_uniform_noise(x):
    """ 对输入添加均匀噪声（范围 [0, 1/256]），用于去量化 """
    noise = torch.rand_like(x) / 256.0
    return x + noise

def gaussian_logprob(x, mean, log_sd):
    """ 计算高斯对数概率：
    log p(x) = -0.5 * log(2*pi) - log_sd - 0.5 * ((x-mean)/exp(log_sd))^2
    """
    return -0.5 * math.log(2 * math.pi) - log_sd - 0.5 * ((x - mean) ** 2) / torch.exp(2 * log_sd)

def logabs(x):
    return torch.log(torch.abs(x) + 1e-6)

#####################################
# ActNorm层：利用首个 batch 初始化均值和方差
#####################################

class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
    
    def initialize(self, x):
        with torch.no_grad():
            B, C, H, W = x.shape 
            x_reshaped = x.permute(1, 0, 2, 3).contiguous().view(C, -1) # 将数据 reshape 成 (C, B*H*W)
            mean = x_reshaped.mean(1).view(1, C, 1, 1)
            std = x_reshaped.std(1).view(1, C, 1, 1) + 1e-6 # 使得经过归一化后输出近似零均值单位方差
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / std)
            self.initialized.fill_(1)
    
    def forward(self, x):
        if self.initialized.item() == 0:
            self.initialize(x)
        out = self.scale * (x + self.loc)
        B, C, H, W = x.shape
        logdet = H * W * torch.sum(logabs(self.scale)) #log-determinant：每个空间位置均乘以 scale，故总 log-det = H * W * sum(log|scale|)
        return out, logdet
    
    def reverse(self, x):
        return x / self.scale - self.loc

#####################################
# 1×1 卷积（使用 LU 分解实现可逆 1×1 卷积）
#####################################

class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super(InvConv2dLU, self).__init__()
        # 用随机正交矩阵初始化, 随机矩阵A 的 正交分解 A = Q, R  
        w_init, _ = torch.qr(torch.randn(in_channel, in_channel)) 
        # 进行 LU 分解
        P, L, U = torch.lu_unpack(*torch.lu(w_init))
        self.register_buffer("P", P)
        self.register_buffer("eye", torch.eye(in_channel))
        # 将 L、U 参数化，其中 L 对角线固定为1
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        # 掩码，用于提取 L 的严格下三角和 U 的严格上三角部分
        self.register_buffer("l_mask", torch.tril(torch.ones_like(L), -1))
        self.register_buffer("u_mask", torch.triu(torch.ones_like(U), 1))
        # 对 U 的对角线单独存储：保存符号和 log（绝对值）
        diag_U = torch.diag(U)
        self.register_buffer("sign_U", torch.sign(diag_U))
        self.log_U_diag = nn.Parameter(torch.log(torch.abs(diag_U) + 1e-6))
    
    def calc_weight(self):
        # 重构 L：L_mask * L + I
        L = self.L * self.l_mask + self.eye
        # 重构 U：U_mask * U + diag(sign * exp(log_U_diag))
        U = self.U * self.u_mask + torch.diag(self.sign_U * torch.exp(self.log_U_diag))
        # 重构 1×1 卷积权重
        W = self.P @ L @ U
        return W.view(W.size(0), W.size(1), 1, 1)
    
    def forward(self, x):
        W = self.calc_weight()
        out = F.conv2d(x, W)
        B, C, H, W_size = x.shape
        # log-det: 每个空间位置都有对角元素的乘积贡献
        logdet = H * W_size * torch.sum(self.log_U_diag)
        return out, logdet
    
    def reverse(self, x):
        W = self.calc_weight()
        # 将 4D 权重转换为 2D 后求逆，再恢复 4D 形状
        W_inv = torch.inverse(W.squeeze()).view(W.size(1), W.size(0), 1, 1)
        return F.conv2d(x, W_inv)

#####################################
# Affine Coupling 层
# 这里采用“分支不变”的方式：将输入按通道均分，用后半部分作为条件（与 TF 版一致）
#####################################

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, hidden_channels):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channel, kernel_size=3, padding=1)
        ) # 对最后一层做零初始化（类似于 TensorFlow 中的 Conv2D_zeros）
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    
    def forward(self, x):
        x_a, x_b = x.chunk(2, 1) # # 将输入按通道拆分为两部分，取后半部分作为条件
        h = self.net(x_b)
        log_s, t = h.chunk(2, 1) # h 拆分为 log_scale 和平移 t，通道数各为 x_a 的通道数
        s = torch.sigmoid(log_s + 2) # 使用 sigmoid 激活，并加上偏置2（与 TensorFlow 保持一致）
        y_a = s * (x_a + t)
        y = torch.cat([y_a, x_b], 1)
        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), dim=1) # log-determinant 为 s 的对数和
        return y, logdet
    
    def reverse(self, y):
        y_a, y_b = y.chunk(2, 1)
        h = self.net(y_b)
        log_s, t = h.chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        x_a = y_a / s - t
        x = torch.cat([x_a, y_b], 1)
        return x

#####################################
# FlowStep：依次使用 ActNorm -> 1×1 可逆卷积 -> Affine Coupling
#####################################

class FlowStep(nn.Module):
    def __init__(self, in_channel, hidden_channels):
        super(FlowStep, self).__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.coupling = AffineCoupling(in_channel, hidden_channels)
    
    def forward(self, x):
        out, logdet_act = self.actnorm(x)
        out, logdet_conv = self.invconv(out)
        out, logdet_coup = self.coupling(out)
        return out, (logdet_act + logdet_conv + logdet_coup)
    
    def reverse(self, x):
        x = self.coupling.reverse(x)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)
        return x

#####################################
# Squeeze / Unsqueeze 操作 
#####################################

def squeeze(x): # squeeze将空间信息转换到通道上（channel 倍增×4，H、W 除以2）
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * 4, H // 2, W // 2)

def unsqueeze(x): 
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C // 4, H * 2, W * 2)

#####################################
# Block：由若干 FlowStep 组成，最后做 Split（采用先验网络预测高斯分布参数）
#####################################

class Block(nn.Module):
    def __init__(self, in_channel, n_flow, hidden_channels, split=True):
        super(Block, self).__init__()
        self.flows = nn.ModuleList([FlowStep(in_channel, hidden_channels) for _ in range(n_flow)])
        self.split = split
        if split:
            # 先验网络：输入为 x（未分离部分），输出 split 成 mean 和 log_sd
            self.prior = nn.Conv2d(in_channel // 2, in_channel, kernel_size=3, padding=1)
            nn.init.zeros_(self.prior.weight)
            nn.init.zeros_(self.prior.bias)
        else:
            self.prior = nn.Conv2d(in_channel, 2 * in_channel, kernel_size=3, padding=1)
            nn.init.zeros_(self.prior.weight)
            nn.init.zeros_(self.prior.bias)
    
    def forward(self, x):
        logdet = 0
        for flow in self.flows:
            x, ld = flow(x)
            logdet = logdet + ld
        
        if self.split:
            # 按通道拆分：保留一半用于后续 flow，另一半作为 latent 变量 z
            x, z = x.chunk(2, 1)
            h = self.prior(x)
            mean, log_sd = h.chunk(2, 1)
            log_p = gaussian_logprob(z, mean, log_sd)
            log_p = log_p.view(log_p.size(0), -1).sum(dim=1)
            return x, logdet, log_p, z
        else:
            h = self.prior(torch.zeros_like(x))
            mean, log_sd = h.chunk(2, 1)
            log_p = gaussian_logprob(x, mean, log_sd)
            log_p = log_p.view(log_p.size(0), -1).sum(dim=1)
            return x, logdet, log_p, x
    
    def reverse(self, x, eps=None):
        if self.split:
            h = self.prior(x)
            mean, log_sd = h.chunk(2, 1)
            if eps is None:
                eps = torch.randn_like(mean)
            # 根据先验生成缺失的 latent 部分
            z = mean + torch.exp(log_sd) * eps
            x = torch.cat([x, z], 1)
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x

#####################################
# Glow 模型整体结构（多尺度结构）
#####################################

class Glow(nn.Module):
    def __init__(self, n_blocks, n_flow, in_channel, hidden_channels, image_size):
        super(Glow, self).__init__()
        self.blocks = nn.ModuleList()
        current_channel = in_channel
        self.image_size = image_size
        for i in range(n_blocks - 1):
            # 每个 block 前先做 squeeze，通道数乘 4，再经过 block 后 split 导致通道数加倍
            self.blocks.append(Block(current_channel * 4, n_flow, hidden_channels, split=True))
            current_channel = current_channel * 2
        self.blocks.append(Block(current_channel * 4, n_flow, hidden_channels, split=False))
    
    def forward(self, x):
        x = x - 0.5 # 数据预处理：原始输入范围 [0,1] 变为 [-0.5, 0.5]，并加入均匀噪声
        x = add_uniform_noise(x)
        logdet_total = 0
        log_p_total = 0
        z_outs = []
        for block in self.blocks:
            x = squeeze(x)
            x, ld, log_p, z = block(x)
            logdet_total = logdet_total + ld
            log_p_total = log_p_total + log_p
            z_outs.append(z)
        B, C, H, W = x.shape #注意：计算 bits/dim 时，需加上去量化常数项：-D*log(1/256), D = total_pixels
        total_pixels = C * H * W 
        c = - total_pixels * math.log(1 / 256.) # NLL（单位为 bits/dim）
        nll = -(log_p_total + logdet_total + c) / (total_pixels * math.log(2))
        return nll, z_outs

    def reverse(self, z_list, temperature=1):
        x = z_list[-1] * temperature # # 注意这里对 latent 乘以温度因子，温度越高越多样
        for i, block in enumerate(reversed(self.blocks)):
            if i == 0:
                x = block.reverse(x, eps=None)
            else:
                eps = z_list[-(i+1)] * temperature
                x = block.reverse(x, eps)
            x = unsqueeze(x) # 逆过程 unsqueeze 的位置必须与 forward 中 squeeze 对应
        x = (x + 0.5).clamp(0, 1)
        return x

#####################################
# 训练与采样部分
#####################################

# model = Glow(n_blocks, n_flows, in_channel, hidden_channels, image_size).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
#     for x, _ in progress_bar:
#         x = x.to(device)
#         optimizer.zero_grad()
#         nll, _ = model(x)
#         loss = nll.mean()
#         loss.backward()
#         optimizer.step()
#         progress_bar.set_postfix(loss=loss.item())
#         total_loss += loss.item()
#     progress_bar.set_postfix(total_loss=total_loss)
#     # 每 3 个 epoch 采样生成图像
#     if epoch % 3 == 0:
#         model.eval()
#         with torch.no_grad():
#             z_samples = []
#             current_channel = in_channel
#             # 根据每个 block 的 latent 大小构造随机采样
#             for i in range(1, n_blocks + 1):
#                 if i < n_blocks:
#                     channels = current_channel * (2 ** i)
#                 else:
#                     channels = current_channel * (2 ** (i + 1))
#                 spatial_size = image_size // (2 ** i)
#                 z_sample = torch.randn(64, channels, spatial_size, spatial_size).to(device)
#                 z_samples.append(z_sample)
#             samples = model.reverse(z_samples)
#             torchvision.utils.save_image(
#                 samples, 
#                 os.path.join(folder_path, f"epoch_{epoch}.png"),
#                 nrow=8,
#                 normalize=True)
#     if epoch % 50 == 0:
#         torch.save(model.state_dict(), os.path.join(folder_path, f"glow_model_epoch_{epoch}.pth"))

#推理模型
output_path = './GLOW_reconstructed_samples_output/'
model_path = './Glow_v6_hidden=256_bflow=3-30-[-0.5,+0.5]_fmnist/glow_model_epoch_50.pth'
def infer_reconstruct_samples(model_path, dataset, output_path, num_samples=16, image_size=32, device='cuda'):
    os.makedirs(output_path, exist_ok=True)
    
    # 加载模型
    model = Glow(n_blocks, n_flows, in_channel, hidden_channels, image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 生成样本
    with torch.no_grad():
        z_samples = []
        current_channel = in_channel
        for i in range(1, n_blocks + 1): # 根据每个 block 的 latent 大小构造随机采样
            if i < n_blocks:
                channels = current_channel * (2 ** i)
            else:
                channels = current_channel * (2 ** (i + 1))
            spatial_size = image_size // (2 ** i)
            z_sample = torch.randn(64, channels, spatial_size, spatial_size).to(device)
            z_samples.append(z_sample)
        samples = model.reverse(z_samples)
        torchvision.utils.save_image(samples, os.path.join(output_path, 'generated_samples2.png'), nrow=8, normalize=True)

    # 获取数据集样本
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    
    # 进行重构
    with torch.no_grad():
        _, encoded = model.forward(images.to(device))
        reconstructed = model.reverse(encoded).cpu()
    
    # 拼接原图和重构图
    comparison = torch.cat([images.view(-1, 1, image_size, image_size).cpu(), reconstructed], dim=0)
    torchvision.utils.save_image(comparison, os.path.join(output_path, 'reconstructed_samples2.png'), nrow=16, normalize=True)
    print(f"samples saved")

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
infer_reconstruct_samples(model_path, test_dataset, output_path)
