import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = './niceS_mnist_block100_hf=pixels---' # 
os.makedirs(folder_path, exist_ok=True)

batch_size = 128
epochs = 501
learning_rate = 3e-4 #3e-4， 1e-3
in_features = 28 * 28    # MNIST 展平后 784 维
hidden_features = in_features
num_blocks = 100

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,)),])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class NICEBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(NICEBlock, self).__init__()
        self.in_features = in_features
        self.m_net = nn.Sequential( # 输入为 in_features/2，输出为 in_features/2，用于计算平移量
            nn.Linear(in_features // 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1) # x 的形状：(batch, in_features), 均分为两部分
        shift = self.m_net(x1)
        y1 = x1
        y2 = x2 + shift
        y = torch.cat([y1, y2], dim=1)
        log_det = torch.zeros(x.size(0), device=x.device) # 加性耦合变换体积保持，log_det = 0
        return y, log_det

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        shift = self.m_net(y1)
        x1 = y1
        x2 = y2 - shift
        x = torch.cat([x1, x2], dim=1)
        return x

class NICE(nn.Module):
    def __init__(self, num_blocks, in_features, hidden_features, device):
        super(NICE, self).__init__()
        self.blocks = nn.ModuleList()
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(in_features, device=device),
            covariance_matrix=torch.eye(in_features, device=device)
        ) # 定义先验：标准多元正态分布
        for _ in range(num_blocks):
            self.blocks.append(NICEBlock(in_features, hidden_features))
        self.log_scale = nn.Parameter(torch.zeros(in_features)) # 添加全局缩放因子

    def forward(self, x): # 前向传播：将数据 x 映射到潜变量 z，并累计 log_det（此处均为 0）。
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for i, block in enumerate(self.blocks): # 根据块编号交替改变分块顺序
            if i % 2 == 0:
                x_a, x_b = x.chunk(2, dim=1)
            else:
                x_b, x_a = x.chunk(2, dim=1)
            x_cat = torch.cat([x_a, x_b], dim=1)
            x, log_det = block(x_cat)
            log_det_total += log_det  # 加性耦合的 log_det 恒为 0
        log_det_total += torch.sum(self.log_scale) # 加入缩放层的 log-det：这里每个样本贡献 sum(log|scale|) 
        x = x * torch.exp(self.log_scale) # 同时将输出进行缩放
        return x, log_det_total

    def inverse(self, z): #逆变换：将潜变量 z 映射回数据 x。
        z = z / torch.exp(self.log_scale) # 逆缩放
        for i, block in enumerate(reversed(self.blocks)):
            y_a, y_b = z.chunk(2, dim=1)
            if i % 2 == 0:
                z = block.inverse(torch.cat([y_a, y_b], dim=1))
            else:
                z = block.inverse(torch.cat([y_b, y_a], dim=1))
        return z

    def log_prob(self, x):
        z, log_det = self.forward(x) 
        log_pz = self.prior.log_prob(z) # 其中 p(z) 为先验分布，log_det 对于加性耦合恒为 0。
        return log_pz + log_det # 计算 x 的对数概率：log p(x) = log p(z) + log_det，

def sample(model, num_samples, in_features, device):
    z = torch.randn(num_samples, in_features).to(device)
    with torch.no_grad():
        x = model.inverse(z)
    return x # 从标准正态分布采样, 先从先验采样，再经过逆变换生成数据

#训练模型
# model = NICE(num_blocks, in_features, hidden_features, device).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0.0
#     progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
#     for batch_idx, (data, _) in enumerate(progress_bar):
#         data = data.view(data.size(0), -1).to(device) # # 将 MNIST 图像展平为向量（28x28 -> 784）
#         optimizer.zero_grad()
#         log_prob = model.log_prob(data)
#         loss = -log_prob.mean()  # 最大化似然
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
#     if (epoch) % 3 == 0:  # 生成样本
#         samples = sample(model, 64, 28*28, device).view(-1, 1, 28, 28).cpu() # (-1, 3, 32, 32), (-1, 1, 28, 28)
#         torchvision.utils.save_image(samples, os.path.join(folder_path, f'samples_epoch{epoch+1}.png'),nrow=8, normalize=True)
#         if( epoch) % 9 == 0:
#             torch.save(model.state_dict(), os.path.join(folder_path, 'best_model.pth'))

#推理模型
model_path = './niceS_mnist_block100_hf=pixels---/best_model.pth'
def infer_reconstruct_samples(model_path, dataset, output_path, num_samples=16, image_size=(28, 28), device='cuda'):
    os.makedirs(output_path, exist_ok=True)
    
    # 加载模型
    model = NICE(num_blocks=100, in_features=28*28, hidden_features=28*28, device=device).to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 生成样本
    with torch.no_grad():
        samples = sample(model, 64, 28*28, device).view(-1, 1, *image_size).cpu()
    torchvision.utils.save_image(samples, os.path.join(output_path, 'niceS_mnist_block100_hf=pixels-generated_samples3.png'), nrow=8, normalize=True)
    
    # 获取数据集样本
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.view(num_samples, -1).to(device)
    
    # 进行重构
    with torch.no_grad():
        encoded, _ = model.forward(images)
        reconstructed = model.inverse(encoded).view(-1, 1, *image_size).cpu()
    
    # 拼接原图和重构图
    comparison = torch.cat([images.view(-1, 1, *image_size).cpu(), reconstructed], dim=0)
    torchvision.utils.save_image(comparison, os.path.join(output_path, 'niceS_mnist_block100_hf=pixels-reconstructed_samples3.png'), nrow=16, normalize=True)
    print(f"samples saved")

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
infer_reconstruct_samples(model_path, test_dataset, './reconstructed_samples_output/')