import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os

# 创建保存图像和模型的文件夹
os.makedirs('./ddpm_results/pre-trained-model', exist_ok=True)

# 预处理数据：标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 下载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super(DDPM, self).__init__()
        self.model = model
        self.timesteps = timesteps

        # 预计算噪声
        self.beta = torch.linspace(1e-4, 0.02, timesteps)  # 可调的beta
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward(self, x_0, t):
        """
        通过正向扩散过程添加噪声
        """
        noise = torch.randn_like(x_0)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise

    def reverse_process(self, x_t, t):
        """
        反向过程，使用UNet模型生成恢复的图像
        """
        model_out = self.model(x_t)
        return model_out

# 创建DDPM实例
model = UNet()
ddpm = DDPM(model)

# 选择优化器
optimizer = optim.Adam(ddpm.parameters(), lr=1e-4)

def train(ddpm, trainloader, optimizer, epochs=10):
    ddpm.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, _) in enumerate(trainloader):
            images = images.cuda()

            # 随机选择时间步t
            t = torch.randint(0, ddpm.timesteps, (images.size(0),), device=images.device)

            # 扩散过程
            noisy_images = ddpm.forward(images, t)

            # 反向过程
            optimizer.zero_grad()
            predicted = ddpm.reverse_process(noisy_images, t)

            # 计算损失（均方误差）
            loss = F.mse_loss(predicted, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader)}')

        # 每5个epoch保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(ddpm.state_dict(), f'./ddpm_results/pre-trained-model/ddpm_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')

# 开始训练
train(ddpm, trainloader, optimizer, epochs=10)

# 生成并保存图像
def generate_images(ddpm, num_images=10, save_dir='./ddpm_results'):
    ddpm.eval()
    with torch.no_grad():
        # 生成噪声
        x_t = torch.randn(num_images, 3, 32, 32).cuda()
        
        for t in reversed(range(ddpm.timesteps)):
            x_t = ddpm.reverse_process(x_t, torch.full((num_images,), t, device=x_t.device))

        # 将图像反向映射到[0, 1]并保存
        x_t = x_t.cpu().clamp(-1, 1)  # 图像应该在[-1, 1]之间
        grid = vutils.make_grid(x_t, nrow=5)

        # 保存图像
        img_path = os.path.join(save_dir, f'generated_image_{torch.randint(1000, (1,)).item()}.png')
        vutils.save_image(grid, img_path)
        print(f"Image saved to {img_path}")

# 生成并保存图像
generate_images(ddpm)

