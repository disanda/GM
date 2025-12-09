import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到 [-1, 1]
])

# 超参数
latent_dim = 128
lr = 0.0002
num_epochs = 50
data_dim = 28*28
batch_size = 128
os.makedirs("gan_results3-convDG", exist_ok=True) # 确保 results 文件夹存在

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 设置GPU id 默认为0 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim=latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            # 输入是 latent_dim x 1 x 1，输出是 feature_dim*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            
            # feature_dim*8 x 4 x 4 -> feature_dim*4 x 8 x 8
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            
            # feature_dim*4 x 8 x 8 -> feature_dim*2 x 14 x 14
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            
            # feature_dim*2 x 16 x 16 -> 1 x 28 x 28
            nn.ConvTranspose2d(feature_dim * 2, 1, kernel_size=6, stride=2, padding=1, bias=False),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, feature_dim=latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            # 输入是 1 x 28 x 28 -> feature_dim x 14 x 14
            nn.Conv2d(1, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # feature_dim x 14 x 14 -> feature_dim*2 x 7 x 7
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # feature_dim*2 x 7 x 7 -> feature_dim*4 x 3 x 3
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # feature_dim*4 x 3 x 3 -> 1 (分类概率)
            nn.Conv2d(feature_dim * 4, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率值
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 展平
        return self.model(img_flat)


# 初始化生成器和判别器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator2().to(device)

# 损失函数
criterion = nn.BCELoss()

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 生成真实数据分布（例如二维正态分布）
def real_data_sampler(batch_size):
    return torch.randn(batch_size, data_dim) * 0.5 + torch.tensor([2, 2])  # 偏移中心 (2, 2)

for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(train_loader):
        batch_size = real_imgs.size(0)
        
        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ---------------------
        #  训练判别器
        # ---------------------
        real_imgs = real_imgs.to(device)
        #z = torch.randn(batch_size, latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(z).detach()  # 假图像，不更新生成器
        # print(fake_imgs.shape)
        # print(discriminator(real_imgs).shape)
        # print(fake_labels.shape)
        
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        d_loss = real_loss + fake_loss
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ---------------------
        #  训练生成器
        # ---------------------
        #z = torch.randn(batch_size, latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real_labels)  # 目标是骗过判别器
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
    # 打印损失
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # 每 3 个 epoch 保存生成图像
    if (epoch + 1) % 3 == 0:
        with torch.no_grad():
            #z = torch.randn(256, latent_dim).to(device)  # 16x16 的生成图像数量
            z = torch.randn(16*16, latent_dim, 1, 1).to(device)
            samples = generator(z).cpu().numpy()
            samples = (samples + 1) / 2  # 转换回 [0, 1] 范围
            
            # 创建一个16x16的子图
            fig, axs = plt.subplots(16, 16, figsize=(16, 16))
            for ax, img in zip(axs.flatten(), samples):
                ax.imshow(img.squeeze(), cmap='gray')  # 绘制灰度图
                ax.axis('off')  # 隐藏坐标轴
            
            # 保存图像到 results 文件夹
            save_path = f"./gan_results3-convDG/epoch_{epoch + 1}.png"
            plt.subplots_adjust(wspace=0, hspace=0)  # 去掉子图间距
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # 关闭图像以释放内存