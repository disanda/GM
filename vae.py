import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context #windows下载mnist出错需要设置

import os
folder_path = './vae_results' # 指定文件夹路径
if not os.path.exists(folder_path): # 检查文件夹是否存在，如果不存在则创建
    os.makedirs(folder_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 设置GPU id 默认为0 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(256, latent_dim)  # 对数方差

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 用 Sigmoid 将输出值压缩到 [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """使用重参数化技巧生成潜在变量 z"""
        std = torch.exp(0.5 * logvar)  # 标准差
        eps = torch.randn_like(std)    # 标准正态分布的随机噪声
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # 重构误差（BCE）
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div

# 超参数
latent_dim = 32  # 潜在空间维度
input_dim = 28 * 28  # MNIST 图像大小
batch_size = 64
epochs = 50
lr = 0.0005

# 数据加载器
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([transforms.ToTensor()])  # 只做 ToTensor 修改数据预处理，去除归一化到 [-1, 1]
dataset = torchvision.datasets.MNIST(root='./data/mnist/', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)

for epoch in range(epochs):
    vae.train()
    total_loss = 0

    for images, _ in dataloader:
        # 预处理数据
        images = images.view(-1, input_dim).to(device)

        # 前向传播
        recon_images, mu, logvar = vae(images)

        # 计算损失
        loss = vae_loss(recon_images, images, mu, logvar)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(dataloader.dataset):.4f}")

    if epoch%3 ==0:
        vae.eval()
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(16*16, latent_dim).to(device)  # 16x16 = 256 个样本
            generated_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()

            # 创建 16x16 的网格
            grid = torchvision.utils.make_grid(generated_images, nrow=16, normalize=True)

            # 保存生成的图像为文件
            torchvision.utils.save_image(grid, './vae_results/generated_images_%d.png'%epoch, normalize=True)

            # 显示图像
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')  # 去除坐标轴
            plt.show()

