import torch
import torchvision
from torch import nn
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = './real_nvp_fmnist_block=100_bs128_hidden=in_norm'
os.makedirs(folder_path, exist_ok=True)

num_blocks = 100    # 块数量, 需要足够多, 100层以上
in_features = 28 * 28 #28 * 28  32*32*3 
hidden_features =  in_features # 增大隐藏层维度，不宜过大或过小
batch_size = 128        # 增大批大小
learning_rate = 3e-4    # 调整学习率
epochs = 600         

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,)),]) # [0,1] > [-1,1]

#train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   

#test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RealNVPBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        
        self.net_s = nn.Sequential( # s网络：输出缩放因子（无最后激活）
            nn.Linear(in_features//2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features//2)
        )
        
        self.net_t = nn.Sequential( # t网络：输出平移因子（无最后激活）
            nn.Linear(in_features//2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features//2)
        )

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=1) # (batch_size, features) => (batch_size, features//2)*2
        s = torch.tanh(self.net_s(x_a))*0.2 #计算缩放和平移因子 tanh:[-1,1] 0.2*: [-0.2, 0.2]
        y_a = x_a
        y_b = self.net_t(x_a) + x_b * torch.exp(s) # 变换特征, t = self.net_t(x_a)
        log_det = s.sum(dim=1)  #计算对数行列式
        return torch.cat([y_a, y_b], dim=1), log_det

    def backward(self, y): # 逆变换
        y_a, y_b = y.chunk(2, dim=1)
        s = torch.tanh(self.net_s(y_a))*0.2 #计算缩放和平移因子
        x_a = y_a
        x_b = (y_b - self.net_t(y_a)) * torch.exp(-s) #  t = self.net_t(y_a)
        return torch.cat([x_a, x_b], dim=1)

class RealNVP(nn.Module):
    def __init__(self, num_blocks, in_features, hidden_features):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(in_features, device=device),  # 均值向量
            covariance_matrix=torch.eye(in_features, device=device)  # 协方差矩阵
        )# 使用多维正态分布作为先验
        
        for _ in range(num_blocks):
            self.blocks.append(RealNVPBlock(in_features, hidden_features)) # # 交替改变分块方向

    def forward(self, x):
        log_det_total = torch.zeros(x.size(0), device=x.device)

        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x_a, x_b = x.chunk(2, dim=1)
            else:
                x_b, x_a = x.chunk(2, dim=1)
            y, log_det = block(torch.cat([x_a, x_b], dim=1))
            log_det_total += log_det
            x = y  # 不再翻转 x
        return x, log_det_total

    def backward(self, z):
        for i, block in enumerate(reversed(self.blocks)):
            y_a, y_b = z.chunk(2, dim=1)
            if i % 2 == 0:
                z = block.backward(torch.cat([y_a, y_b], dim=1))
            else:
                z = block.backward(torch.cat([y_b, y_a], dim=1))  # 交换 y_a, y_b
        return z

    def log_prob(self, x):
        z, log_det = self.forward(x) #z = torch.clamp(z, min=-1e6, max=1e6)
        prior_logprob = self.prior.log_prob(z)  # 计算多维正态分布的对数概率
        return prior_logprob + log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))  #从多维正态分布中采样,  形状为 (num_samples, in_features)
        return self.backward(z)

#训练模型
# model = RealNVP(num_blocks, in_features, hidden_features).to(device) # 初始化模型
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 学习率调度
# best_loss = float('inf')
# for epoch in range(epochs): # 训练循环
#     model.train()
#     total_loss = 0.0
#     progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    
#     for x, _ in progress_bar:
#         x = x.view(-1, in_features).to(device) # (batch_size, features)
#         loss = -model.log_prob(x).mean() # 计算负对数似然
        
#         optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
#         optimizer.step()
        
#         total_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())
    
#     scheduler.step()
#     avg_loss = total_loss / len(train_loader)
    
#     model.eval()  # 验证
#     with torch.no_grad():
#         val_loss = 0.0
#         for x, _ in test_loader:
#             x = x.view(-1, in_features).to(device)
#             val_loss += -model.log_prob(x).mean().item()
#         val_loss /= len(test_loader)
#     print(f'Epoch {epoch+1} | Train Loss: {avg_loss:.3f} | Val Loss: {val_loss:.3f}')
    
#     if val_loss < best_loss: #保存最佳模型和生成样本
#         torch.save(model.state_dict(), os.path.join(folder_path, 'best_model.pth'))
    
#     if (epoch) % 3 == 0:  # 生成样本
#         with torch.no_grad():
#             samples = model.sample(64).view(-1, 1, 28, 28).cpu() # (-1, 3, 32, 32), (-1, 1, 28, 28)
#             torchvision.utils.save_image(samples, os.path.join(folder_path, f'samples_epoch{epoch+1}.png'),nrow=8, normalize=True)

#推理模型
output_path = './real_NVP_reconstructed_samples_output/'
model_path = './real_nvp_fmnist_block=100_bs128_hidden=in_norm-/best_model.pth'
def infer_reconstruct_samples(model_path, dataset, output_path, num_samples=16, image_size=(28, 28), device='cuda'):
    os.makedirs(output_path, exist_ok=True)
    
    # 加载模型
    model = RealNVP(num_blocks, in_features, hidden_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 生成样本
    with torch.no_grad():
        samples = model.sample(64).view(-1, 1, *image_size).cpu() 
        torchvision.utils.save_image(samples, os.path.join(output_path, 'real_nvp_fmnist_block=100_bs128_hidden=in_norm-generated_samples.png'), nrow=8, normalize=True)
    
    # 获取数据集样本
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.view(num_samples, -1).to(device)
    
    # 进行重构
    with torch.no_grad():
        encoded, _ = model.forward(images)
        reconstructed = model.backward(encoded).view(-1, 1, *image_size).cpu()
    
    # 拼接原图和重构图
    comparison = torch.cat([images.view(-1, 1, *image_size).cpu(), reconstructed], dim=0)
    torchvision.utils.save_image(comparison, os.path.join(output_path, 'real_nvp_fmnist_block=100_bs128_hidden=in_norm-reconstructed_samples.png'), nrow=16, normalize=True)
    print(f"samples saved")

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
infer_reconstruct_samples(model_path, test_dataset, output_path)
