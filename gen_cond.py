import math

import torch

from model.diffusion.Unet1D import Unet1D_crossatt
# from model.diffusion.Unet1D_SNN import Unet1D_crossatt
from model.diffusion.diffusion import GaussianDiffusion1D
from evaluate.evaluate_utils import *
from dataset import *
from torch.utils.data import DataLoader
import os
from pathlib import Path

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
Batch_Size = 1
norm_type = 'none' # recommend 1-1
data_state='outer3' # SQ_M: normal,inner(1,2,3),outer(1,2,3) SQV: NC,IF(1,2,3),OF(1,2,3)

length=4000
data_num=10
patch = 8 if Batch_Size >= 64 else 4
cond_np=None

# diffusion para
dif_object = 'pred_v'
beta_schedule= 'linear' ## linear
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000
loss_type='huber' # l1,l2,huber

#use gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

path_name = '2025_0603_172303'
index='TUM_COND'
epoch=400

datasets, data_np, cond = build_dataset(
    dataset_type=index,
    normlizetype=norm_type,
    ch=5,
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
val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True, num_workers=8,drop_last=True)
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

model = Unet1D_crossatt(
    dim=32,
    num_layers=4,
    dim_mults=(1, 2, 4, 8),
    context_dim= length,
    channels=1,
    use_crossatt=False, # True for VGCDM, False for DDPM
)

model.to(device)

model_ckpt = torch.load('/data/jionkim/VGCDM/output/' + index + '/' + path_name +'/output/model_' + str(epoch) + '.ckpt')
model.load_state_dict(model_ckpt)

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
save_path = '/data/jionkim/VGCDM/output/' + index + '/' + path_name + '/output/'
print("output_path:{}".format(save_path))
Path(save_path).mkdir(parents=True, exist_ok=True)

for j in range (0, 16):
    all_sampled_seqs = []
    for i in range (0, 12):
        print(i)
        idx = torch.Tensor([i])
        sampled_seq = diffusion.sample(batch_size=1, cond=idx).detach().cpu().numpy()
        all_sampled_seqs.append(sampled_seq)

    final_data = np.array(all_sampled_seqs)
    np.save(save_path + 'final_result_' + str(j).zfill(2) + '.npy', arr=final_data)