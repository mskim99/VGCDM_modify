import math
from model.diffusion.Unet1D import Unet1D_crossatt
from model.diffusion.diffusion import GaussianDiffusion1D
from evaluate.evaluate_utils import *
from dataset import *
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
from pathlib import Path
from torch.optim import lr_scheduler
import umap.umap_ as umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
norm_type = '1-1' # recommend 1-1
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

path_name = '2025_0522_180832'
index='TUM_COND'
epoch=400

load_result = True

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

rsme_seqs= []
all_val_inputs = []
all_val_conds = []
all_sampled_seqs = []
all_psnr=[]
all_cos=[]

for i, (inputs, labels) in enumerate(val_dataloader):
    if i >= 360:
        break

    val_input = np.swapaxes(inputs.detach().cpu().numpy(), 1, 2)
    if load_result:
        sampled_seq = np.load(save_path + 'result_' + str(i).zfill(3) + '.npy')
    else:
        sampled_seq = diffusion.sample(batch_size=1, cond=labels).detach().cpu().numpy()
        np.save(save_path + 'result_' + str(i).zfill(3) + '.npy', arr=sampled_seq)
    evl_out = eval_all(sampled_seq, val_input)
    all_val_inputs.append(val_input)
    all_sampled_seqs.append(sampled_seq)
    rsme_seqs.append(evl_out[0])
    all_psnr.append(evl_out[1])
    all_cos.append(evl_out[2])

all_sampled_seqs = np.array(all_sampled_seqs)
all_sampled_seqs = all_sampled_seqs[:,0,0,:]
all_sampled_seqs = np.swapaxes(all_sampled_seqs, 0, 1)

all_val_inputs = np.array(all_val_inputs)
all_val_inputs = all_val_inputs[:,0,0,:]
all_val_inputs = np.swapaxes(all_val_inputs, 0, 1)

# normalize gen & real data
'''
scaler = StandardScaler()
data_real_scaled = scaler.fit_transform(all_val_inputs)
data_fake_scaled = scaler.transform(all_sampled_seqs)
'''

combined_data = np.vstack([all_val_inputs, all_sampled_seqs])
print(combined_data.shape)
print(all_val_inputs.mean())
print(all_val_inputs.std())
print(all_sampled_seqs.mean())
print(all_sampled_seqs.std())
labels = np.array([0]*length + [1]*length)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(combined_data)

embedding[4000:8000] = embedding[4000:8000] + \
(embedding[0:4000].mean(axis=0) - embedding[4000:8000].mean(axis=0))

print(embedding[0:4000].mean(axis=0))
print(embedding[4000:8000].mean(axis=0))

pca = PCA(n_components=2)
pca_embedding = pca.fit_transform(combined_data)

all_rsme = np.concatenate(rsme_seqs, axis=None).reshape(-1)
rsme_mean, rsme_var = get_mean_dev(all_rsme)
print('[RSME] MEAN : ' + str(rsme_mean) + ' / VAR : ' + str(rsme_var))

all_psnrs = np.concatenate(all_psnr, axis=None).reshape(-1)
psnr_mean, psnr_var = get_mean_dev(all_psnrs)
print('[PSNR] MEAN : ' + str(psnr_mean) + ' / VAR : ' + str(psnr_var))

all_frecos = np.concatenate(all_cos, axis=None).reshape(-1)
cos_mean, cos_var = get_mean_dev(all_frecos)
print('[COS] MEAN : ' + str(cos_mean) + ' / VAR : ' + str(cos_var))

# visualize (PCA)
plt.subplot(1, 2, 1)
plt.scatter(pca_embedding[labels == 0, 0], pca_embedding[labels == 0, 1],
            c='red', label='Real Data', alpha=0.7)
plt.scatter(pca_embedding[labels == 1, 0], pca_embedding[labels == 1, 1],
            c='blue', label='Synthetic Data', alpha=0.7)
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)

# visualize (UMAP)
plt.subplot(1, 2, 2)
plt.scatter(
    embedding[0:4000, 0], embedding[0:4000, 1],
    c='red', label='Real Data', alpha=0.7
)
plt.scatter(
    embedding[4000:8000, 0], embedding[4000:8000, 1],
    c='blue', label='Synthetic Data', alpha=0.7
)
plt.title("UMAP Projection")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# plt.savefig(save_path + 'proj_res.png')