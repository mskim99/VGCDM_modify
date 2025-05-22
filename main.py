import torch
import numpy as np
import math
# from diffusion.diffusion_1d import Unet1D, GaussianDiffusion1D
from model.diffusion.Unet1D import Unet1D_crossatt,Unet1D
from model.diffusion.diffusion import GaussianDiffusion1D
from dataset import *
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from evaluate import *
import datetime
import os
from pathlib import Path
from utils.logger import create_logger
from torch.optim import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

def get_mean_dev(y):
    b=y.shape[0]
    y=y.reshape(1,b)
    mean=np.mean(y,axis=1)
    variance = np.var(y,axis=1)
    # print(mean,variance)
    return mean, variance

def default_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Specify output directory here
output_dir = "./output"
default_dir(output_dir)
Batch_Size = 32
norm_type = '1-1' # recommend 1-1
index='TUM' # add _M for (inputs, labels,context) else (input, labels)
data_state='outer3' # SQ_M: normal,inner(1,2,3),outer(1,2,3) SQV: NC,IF(1,2,3),OF(1,2,3)

length=4000
data_num=10
patch = 8 if Batch_Size >= 64 else 4


cond_np=None
# time
new_dir=default_dir(os.path.join(output_dir,index))
cur_time=datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
time_out_dir=default_dir(os.path.join(new_dir,cur_time))
logger = create_logger(output_dir=time_out_dir,  name=f"{index}.txt")
logger.info("index:{},norm_type:{};data_num:{}"
                .format(index,norm_type,data_num))


# diffusion para
dif_object = 'pred_v'
beta_schedule= 'linear' ## linear
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000
epochs = 401
loss_type='huber' # l1,l2,huber

logger.info("dif_object:{},beta_schedule:{},beta:{}-{};epochs:{};diffusion time step:{};loss type:{}"
                .format(dif_object,beta_schedule,beta_start,beta_end,epochs,timesteps,loss_type))

#use gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

if index=="TUM":
    datasets, data_np, cond = build_dataset(
        dataset_type='TUM',
        normlizetype=norm_type,
        ch=5,
        data_num=data_num,
        length=length,
    )
elif index=="TUM_COND":
    datasets, data_np, cond = build_dataset(
        dataset_type='TUM_COND',
        normlizetype=norm_type,
        ch=5,
    )
else:
    raise ('unexpected data index, please choose data index form TUM, TUM_COND')

# plot origin data
logger.info("condition:{}".format(cond))
'''
ori_path=os.path.join(time_out_dir,cur_time+cond+'_ori.png')
if '_M' in index:
    plot_np(y=data_np,z=cond_np,patch=patch,path= ori_path,show_mode=False)
else:
    plot_np(y=data_np,patch=patch,path= ori_path,show_mode=False)
'''
train_dataset,val_dataset=get_loaders(
    datasets['train'],
    val_ratio=0.3,
    batch_size=Batch_Size,
    with_test=False,
    with_label=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8,drop_last=True)
logger.info("train_num:{};val_num:{}".format(len(train_dataset),len(val_dataset)))

# define beta schedule
def linear_beta_schedule(timesteps,beta_start = 0.0001,beta_end = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

if beta_schedule =='cosine':
    betas = cosine_beta_schedule(timesteps)
else:
    betas = linear_beta_schedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end)

#define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1",context=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    # predicted_noise = denoise_model(x_noisy, t, context=context)
    predicted_noise = denoise_model(x_noisy, t, x_self_cond=context)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

model = Unet1D_crossatt(
    dim=32,
    num_layers=4,
    dim_mults=(1, 2, 4, 8),
    context_dim= length,
    channels=1,
    use_crossatt=False, # True for VGCDM, False for DDPM
)

model.to(device)

optimizer = AdamW(params=model.parameters(),lr=1e-5,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# optimizer = Adam(model.parameters(), lr=1e-4)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = length,
    timesteps = timesteps, #1000
    objective = dif_object, #'pred_v'
    beta_schedule=beta_schedule,#'linear'
    auto_normalize=False
)

diffusion.to(device)

# save_model path check
save_path = time_out_dir + '/output/'
logger.info("output_path:{}".format(save_path))
Path(save_path).mkdir(parents=True, exist_ok=True)

for epoch in range(epochs):
    # for step, (inputs, labels,context) in enumerate(train_dataloader):
    for step, (inputs, labels) in enumerate(train_dataloader):
      optimizer.zero_grad()
      inputs = torch.swapaxes(inputs, 1, 2)
      batch_size = inputs.shape[0]
      batch = inputs.to(device).float()
      context = labels.to(device).float()

      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
      loss = p_losses(model, batch, t, loss_type=loss_type, context=None)
      if step % 100 == 0:
        learning_rate = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info("Epoch:{}, Loss:{}; Mem:{} MB, Lr:{}".format(epoch,round(loss.item(), 4),round(memory_used),learning_rate))

      loss.backward()
      optimizer.step()
    scheduler.step()

    if epoch % 20 == 0:
        # Save checkpoint
        save_name = 'model_' + str(epoch).zfill(3) + '.ckpt'
        torch.save(model.state_dict(), save_path + save_name)

        sampled_seq = diffusion.sample(batch_size=1, cond=None)
        np.save(save_path + 'output_' + str(epoch).zfill(3) + '.npy', arr=sampled_seq.cpu().numpy())