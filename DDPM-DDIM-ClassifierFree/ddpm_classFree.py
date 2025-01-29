import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 设置GPU id 默认为0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 501
batch_size = 64
image_size = 28 # cifar10  = 32 or fmnist 28
timesteps = 1000 # fmnist, mnist ddim_steps = 300/500, cifar=1000
in_channels = 3 # fmnist, mnist = 1, cifar = 3
ds = 'cifar_step1000_ep501_v1=small_model' # mnist_step500_ep201_v4'# cifar10 or fmnist
exp_name = './ddpm_classFree_'+ds # cifar10 or fmnist
os.makedirs(exp_name+'/pre-trained-model', exist_ok=True) # 创建保存图像和模型的文件夹

train = True # False is inferences via pre-trained model
class_free = True
n_class = 10 # or None 
ddim = False # True
ddim_steps = 50

transform = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
]) 

class Upsample(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.num_groups = num_groups

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest") #        # 上采样
        x = self.conv(x) #        # 卷积 + GroupNorm
        return x  # 激活函数

class Downsample(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x) #卷积 + GroupNorm
        return x # 激活函数

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = F.relu(F.group_norm(self.conv1(x), num_groups=32)) # 第一层卷积 + GroupNorm + 激活
        h = F.relu(F.group_norm(self.conv2(h), num_groups=32))
        return h + self.shortcut(x)  # 残差连接

def timestep_embedding(t, dim, max_period=10000):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim // 2, dtype=torch.float32) / (dim // 2)).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class UNetModel(nn.Module):
    def __init__(self, io_channels=3, model_channels=128, class_num = None):
        super().__init__()
        self.model_channels = model_channels

        self.down_block1 = nn.Conv2d(io_channels, model_channels, kernel_size=3, padding=1) # down blocks
        self.down_block2  =  Downsample(model_channels)
        self.down_block3 = nn.Conv2d(model_channels, model_channels*2, kernel_size=3, padding=1)
        self.down_block4  =  Downsample(model_channels*2)

        self.middle_block = ResidualBlock(model_channels*2, model_channels*2) # middle block
        self.noise_embedding = nn.Linear(model_channels, model_channels*2) # noise block
        if class_free != None:
            self.class_emb = nn.Embedding(class_num, model_channels*2)

        self.up_block1 = Upsample(model_channels*2) #  # up blocks
        self.up_block2 =  nn.Conv2d(model_channels*2, model_channels, kernel_size=3, padding=1)
        self.up_block3 = Upsample(model_channels)
        self.up_block4 =  nn.Conv2d(model_channels, io_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, label=None):
        x1 = F.relu(F.group_norm(self.down_block1(x), num_groups=32)) #Encode-Downsampling 
        x2 = F.relu(F.group_norm(self.down_block2(x1), num_groups=32))
        x3 = F.relu(F.group_norm(self.down_block3(x2), num_groups=32))
        x4 = F.relu(F.group_norm(self.down_block4(x3), num_groups=32))

        middle = self.middle_block(x4) # Middle block
        noise_t = F.relu(self.noise_embedding(timestep_embedding(t,self.model_channels)))
        if label != None:
            c_emb = F.relu(self.class_emb(label))
            middle = middle + noise_t[:, :, None, None] + c_emb[:, :, None, None]
        else:
            middle = middle + noise_t[:, :, None, None]

        x5 = F.relu(F.group_norm(self.up_block1(middle + x4), num_groups=32)) # Decode-Upsampling
        x6 = F.relu(F.group_norm(self.up_block2(x5 + x3 ), num_groups=32)) # 
        x7 = F.relu(F.group_norm(self.up_block3(x6 + x2), num_groups=32)) #  
        out = self.up_block4(x7 + x1) #  

        return out

# model = UNetModel(io_channels=3, model_channels=128) # # 创建网络
# t = torch.randint(0, 500, (32,)).long() 
# x = torch.randn(32, 3, 28, 28)  # # 输入张量
# output = model(x,t) # 前向传播
# print(output.shape)  # 输出

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps 
    beta_start = 0.0003 * scale # 该值不可过小，去燥不充分
    beta_end = 0.03 * scale # 该值不可过小，乱序条纹
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

class GaussianDiffusion:
    def __init__(self, timesteps=1000,):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.betas_cumprod = torch.cumprod(self.betas, axis=0)

        self.alphas = 1. - self.betas # 接近1的数
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None): 
        # 前向加噪过程：forward diffusion (using the nice property): q(x_t | x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor, c = None):
        noise = torch.randn_like(x_start)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(x_start, t, noise = noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t, label = c)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    #DDPM Inference/Reverse
    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, c = None ):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`, 估计值，包含部分残留噪声
        pre_x_0 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * model(x_t, t, c) # pred_noise = model(x_t, t)
        pre_x_0 = torch.clamp(pre_x_0, min=-1., max=1.) # clip_denoised
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(pre_x_0, x_t, t) ## compute predicted mean and variance of p(x_{t-1} | x_t), predict noise using model
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, c = None):
        # denoise_step: sample x_{t-1} from x_t and pred_noise, predict mean and variance
        model_mean, posterior_variance, model_log_variance = self.p_mean_variance(model, x_t, t, c)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))) # no noise when t == 0
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # compute x_{t-1}
        return pred_img

    def sample(self, model: nn.Module, image_size, batch_size=8, channels=3, n_class=None):
        shape = (batch_size, channels, image_size, image_size) # denoise: reverse diffusion
        img = torch.randn(shape, device=device)  #start from pure noise (for each example in the batch), x_T ~ N(0, 1)
        imgs = []
        if n_class != None:
            cur_y = torch.randint(0, n_class, (batch_size,)).to(device) # 随机标签
        else:
            cur_y = None
        for i in tqdm(reversed(range(0, self.timesteps)), desc='ddpm sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, c = cur_y)
            imgs.append(img)
        return imgs

    #DDIM Inference/Reverse
    def ddim_sample(self, model, image_size, ddim_timesteps=100, batch_size=64, channels=3, n_class=10):
        
        if n_class is not None:
            if batch_size % n_class != 0:
                batch_size = n_class * (batch_size // n_class)
                print(f'Adjusted batch_size to {batch_size}')
            images_per_class = batch_size // n_class #生成连续排列的类别标签, 每个类别连续重复 images_per_class 次, 
            cur_y = torch.tensor([x for x in range(n_class) for _ in range(images_per_class)], dtype=torch.long).to(device)
        else: #生成连续排列的类别标签：0,0,...,0, 1,1,...,1, ..., 9,9,...,9
            cur_y = None

        shape = (batch_size, channels, image_size, image_size)
        x_T = torch.randn(shape, device=device)  # start from pure noise
        xs = [x_T]
        c = self.timesteps // ddim_timesteps  # make ddim timestep sequence
        ddim_timestep_seq = torch.tensor(list(range(0, self.timesteps, c))) + 1  # one from first scale to data during sampling
        ddim_timestep_prev_seq = torch.cat((torch.tensor([0]), ddim_timestep_seq[:-1]))  # previous sequence

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='ddpm sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_T.shape)  # 1. get current and previous alpha_cumprod
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, next_t, x_T.shape)

            pred_noise = model(xs[-1], t, cur_y)  # 2. predict noise using model, 模型预测噪声
            pred_x0 = (xs[-1] - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)  # 3. get the predicted x_0, 预测 x_0
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise  # 5. compute "direction pointing to x_t" of formula (12)
            x_t_pre = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt  # 6. compute x_{t-1} of formula (12)
            xs.append(x_t_pre)
            # omit 4. compute variance: "sigma_t(η)" -> see formula (16) / σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        return xs

def denoising_samples(timesteps, generated_images):
    imgs_seq = []
    n_cols = min(16, timesteps)  # 列数最多为 16，但不能超过时间步数
    time_indices = torch.linspace(0, timesteps - 1, steps=n_cols, dtype=torch.long) # 生成均匀分布的时间索引，最后一个固定为 timesteps-1
    for n_row in range(16):  # 假设有 16 行
        for t_idx in time_indices:  # 遍历时间索引列
            img = torch.tensor(generated_images[t_idx][n_row])  # 提取对应时间步和行的图像
            imgs_seq.append(img)
        imgs_seq.append(generated_images[-1][n_row])
    return imgs_seq, n_cols

if train == True: # training ddpm model
    model = UNetModel(io_channels=in_channels,model_channels=128, class_num=n_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    #dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    #dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform) #use MNIST dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)   

    for epoch in range(epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            image_size = int(images.shape[-1])

            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long() 

            if class_free == True:
                loss = gaussian_diffusion.train_losses(model, images, t, labels)
            else:    
                loss = gaussian_diffusion.train_losses(model, images, t, None)

            if step % 200 == 0:
                print("Epoch:", epoch, "Step:", step, "Loss:", loss.item())

            loss.backward()
            optimizer.step()

        if (epoch == 1) or (epoch % 10 == 0):
            with torch.no_grad():
                model.eval()
                
                generated_images = gaussian_diffusion.sample(model, image_size, batch_size=batch_size, channels=in_channels, n_class = n_class)
                torchvision.utils.save_image((generated_images[-1]+ 1)/2, exp_name + '/epoch_%d.png'%epoch, nrow = int(np.sqrt(batch_size)))

                generated_class_images = gaussian_diffusion.ddim_sample(model, image_size, ddim_timesteps=ddim_steps, batch_size=batch_size, channels=in_channels)
                torchvision.utils.save_image((generated_class_images[-1]+ 1)/2, exp_name + '/class_epoch_%d.png'%epoch, nrow = int(np.sqrt(batch_size) if n_class == None else (batch_size // n_class)))

                imgs,n_cols = denoising_samples(timesteps,generated_images)
                imgs = (torch.stack(imgs) + 1) / 2 # [-1, 1] > [0,1]
                torchvision.utils.save_image(imgs, exp_name + '/denoise_epoch=%d.png'%epoch, nrow = n_cols+1)

        if (epoch == 1) or (epoch % 50 == 0):
            torch.save(model.state_dict(), exp_name+'/pre-trained-model/ddpm_epoch_%d.pth'%epoch)
            print(f"Model saved at epoch {epoch}")

else: # inference
    model = UNetModel(io_channels=in_channels, model_channels=128, class_num=10).to(device) #加载预训练模型
    model.load_state_dict(torch.load('./fmnist_ddpm_epoch_150_step300.pth',map_location=device))
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    imgs_time = []

    with torch.no_grad():
        model.eval()
        if ddim == True:
            generated_images = gaussian_diffusion.ddim_sample(model, image_size, ddim_timesteps=ddim_steps, batch_size=batch_size, channels=in_channels)
            imgs,n_cols = denoising_samples(ddim_steps,generated_images)
        else: # ddpm
            generated_images = gaussian_diffusion.sample(model, image_size, batch_size=batch_size, channels=in_channels)
            imgs,n_cols = denoising_samples(timesteps,generated_images)

        torchvision.utils.save_image((generated_images[-1]+ 1)/2, './samples_train=%s_ddim=%s.png'%(train,ddim), nrow = int(np.sqrt(batch_size)))
        imgs = (torch.stack(imgs).reshape(-1, in_channels, image_size, image_size) + 1) / 2 # [-1, 1] > [0,1]
        torchvision.utils.save_image(imgs, './sample_noise_ddim=%s.png'%ddim, nrow = n_cols+1) # denoising samples

# def cosine_beta_schedule(timesteps, s=0.008): # https://arxiv.org/abs/2102.09672
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)
