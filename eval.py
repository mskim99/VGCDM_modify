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

length=1024
data_num=10
patch = 8 if Batch_Size >= 64 else 4
cond_np=None

# diffusion para
dif_object = 'pred_v'
beta_schedule= 'linear' ## linear
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000
epochs = 200
loss_type='huber' # l1,l2,huber

#use gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

datasets, data_np, cond = build_dataset(
    dataset_type='TUM',
    normlizetype=norm_type,
    ch=5,
    data_num=data_num,
    length=length,
)
sr = 12000

# plot origin data
print("condition:{}".format(cond))

train_dataset,val_dataset=get_loaders(
    datasets['train'],
    val_ratio=0.3,
    batch_size=Batch_Size,
    with_test=False,
    with_label=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8,drop_last=True)
print("train_num:{};val_num:{}".format(len(train_dataset),len(val_dataset)))

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
'''
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
'''

model = Unet1D_crossatt(
    dim=32,
    num_layers=4,
    dim_mults=(1, 2, 4, 8),
    context_dim= length,
    channels=1,
    use_crossatt=False, # True for VGCDM, False for DDPM
)

model.to(device)

model_ckpt = torch.load('/data/jionkim/VGCDM/output/TUM/2025_0519_164307/output/model_180.ckpt')
model.load_state_dict(model_ckpt)

optimizer = AdamW(params=model.parameters(),lr=1e-4,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

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
save_path = '/data/jionkim/VGCDM/output/TUM/2025_0519_164307/output/'
print("output_path:{}".format(save_path))
Path(save_path).mkdir(parents=True, exist_ok=True)

rsme_seqs= []
all_val_inputs = []
all_val_conds = []
all_sampled_seqs = []
all_psnr=[]
all_cos=[]

for batch in val_dataloader:

    '''
    val_input = batch[0].to(device).float()
    val_conds = batch[2].to(device).float()
    B=val_input.shape[0]
    '''

    for i in range(16):
        sampled_seq = diffusion.sample(batch_size=1, cond=None)
        # evl_out = eval_all(sampled_seq.detach().cpu().numpy(), val_input.detach().cpu().numpy())
        # all_val_inputs.append(val_input)
        # all_val_conds.append(val_conds)
        np.save(save_path + 'result_' + str(i) + '.npy' ,arr=sampled_seq.cpu().numpy())
        all_sampled_seqs.append(sampled_seq)
        # rsme_seqs.append(evl_out[0])
        # all_psnr.append(evl_out[1])
        # all_cos.append(evl_out[2])

'''
all_val_inputs = torch.cat(all_val_inputs, dim=0)
all_val_conds = torch.cat(all_val_conds, dim=0)
all_sampled_seqs = torch.cat(all_sampled_seqs, dim=0)

all_rsme = np.concatenate(rsme_seqs, axis=None).reshape(-1)
rsme_mean, rsme_var = get_mean_dev(all_rsme)
print('RSME', rsme_mean, rsme_var, all_rsme)

all_psnrs = np.concatenate(all_psnr, axis=None).reshape(-1)
psnr_mean, psnr_var = get_mean_dev(all_psnrs)
print('PSNR', psnr_mean, psnr_var, all_psnrs)

all_frecos = np.concatenate(all_cos, axis=None).reshape(-1)
cos_mean, cos_var = get_mean_dev(all_frecos)
print('fre_cos', cos_mean, cos_var, all_frecos)
print('All_eval', rsme_mean, rsme_var, psnr_mean, psnr_var, cos_mean, cos_var)
logger.info("All_eval-rsme:{}/{},psnr:{}/{},cos:{}/{}".format(rsme_mean, rsme_var, psnr_mean, psnr_var, cos_mean, cos_var))

# for i in range(16):
out_np=all_sampled_seqs[:Batch_Size].cuda().data.cpu().numpy()

val_data_path=os.path.join(time_out_dir,cur_time+cond+'.npy')
val_in=all_val_inputs[:Batch_Size].cuda().data.cpu().numpy()
val_conds_out=all_val_conds[:Batch_Size].cuda().data.cpu().numpy()
out_npy=np.concatenate((out_np,val_in,val_conds_out),axis=1)
'''

# out_npy=np.concatenate((out_np,val_in),axis=1)

# add condition in plot when use cross attention
'''
if '_M' in index:
    val_data_path=os.path.join(time_out_dir,cur_time+cond+'.npy')
    val_output=val_input.cuda().data.cpu().numpy()
    val_conds_out=val_conds.cuda().data.cpu().numpy()
    out_npy=np.concatenate((out_np,val_output,val_conds_out),axis=1)
    np.save(val_data_path,arr=out_npy)
    plot_two_np(x=out_np,
                y=val_output,
                z1=None,
                z2=val_conds_out,
                patch=patch,
                path=os.path.join(time_out_dir,cur_time+cond+'_time.png'),
                show_mode='time',
                sample_rate=sr)
    plot_two_np(x=out_np,
                y=val_input.cuda().data.cpu().numpy(),
                patch=patch,
                path=os.path.join(time_out_dir,cur_time+cond+'_fft.png'),
                show_mode='fft',
                sample_rate=sr)
else:
plot_two_np(x=out_np,
            y=data_np,
            path=os.path.join(time_out_dir, cur_time + cond + '_time.png'),
            patch=patch,
            show_mode='time',
            sample_rate=sr)
plot_two_np(x=out_np,
            y=data_np,
            patch=patch,
            path=os.path.join(time_out_dir, cur_time + cond + '_fft.png'),
            show_mode='fft',
            sample_rate=sr)
'''
#
# df_m,df_v=get_mean_dev(sampled_seq.cuda().data.cpu().numpy())
# print(sampled_seq.shape) # (4, 32, 128)t