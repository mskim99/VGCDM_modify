import numpy as np
import glob

import torch

from evaluate.evaluate_utils import rmse,psnr,fre_cosine,mmd,emd, eval_fid
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

from random import shuffle
import os

def get_mean_dev(y):
    # b=y.shape[0]
    # y=y.reshape(1,b)
    mean=np.mean(y)
    variance = np.var(y)
    # print(mean,variance)
    return mean, variance

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

# gen_paths = glob.glob('/data/jionkim/Test/250603_1/*.npy')
gen_paths = glob.glob('/data/jionkim/VGCDM/output/TUM_COND/2025_0603_012412/output/final_result_*.npy')
gt_paths = glob.glob('/data/jionkim/Test/npy_dir_X/*.npy')
shuffle(gt_paths)

gen_imgs = []
gt_imgs = []

for name in gen_paths:
    img = np.load(name)
    gen_imgs.append(img)

idx = 0
for name in gt_paths:
    if idx >= 16:
        break
    img = np.load(name)
    gt_imgs.append(img)
    idx = idx+1

gen_imgs = np.array(gen_imgs)
gt_imgs = np.array(gt_imgs)

gen_imgs = gen_imgs[:, :, 0, 0, :]
gen_imgs = np.swapaxes(gen_imgs, 0, 1)
gen_imgs = np.concatenate(gen_imgs, axis=1)

# indices = np.random.permutation(gt_imgs.shape[0])
# gt_imgs_shuffle = gt_imgs[indices]
# gt_imgs_shuffle = gt_imgs_shuffle[:16, :]

gen_imgs = np.expand_dims(gen_imgs, 1)
gt_imgs_shuffle = np.expand_dims(gt_imgs, 1)

print(gen_imgs.shape)
print(gt_imgs_shuffle.shape)

# gen_imgs = torch.Tensor(gen_imgs).cuda()
# gt_imgs_shuffle = torch.Tensor(gt_imgs_shuffle).cuda()

rmse_all = rmse(gen_imgs, gt_imgs_shuffle)
psnr_all = psnr(gen_imgs, gt_imgs_shuffle)
fre_cos_all = fre_cosine(gen_imgs, gt_imgs_shuffle)
mmd_all = mmd(gen_imgs, gt_imgs_shuffle)
emd_all = emd(gen_imgs, gt_imgs_shuffle)

rmse_all = np.array(rmse_all)
psnr_all = np.array(psnr_all)
fre_cos_all = np.array(fre_cos_all)
mmd_all = np.array(mmd_all)
emd_all = np.array(emd_all)

rsme_mean, rsme_var = get_mean_dev(rmse_all)
psnr_mean, psnr_var = get_mean_dev(psnr_all)
fre_cos_mean, fre_cos_var = get_mean_dev(fre_cos_all)
mmd_mean, mmd_var = get_mean_dev(mmd_all)
emd_mean, emd_var = get_mean_dev(emd_all)

# print('Calculate FID')
'''
gen_imgs_fid = gen_imgs.cpu().numpy()
gt_imgs_shuffle_fid = gen_imgs.cpu().numpy()
gen_imgs_fid = gen_imgs_fid.swapaxes(0, 2)
gt_imgs_shuffle_fid = gt_imgs_shuffle_fid.swapaxes(0, 2)
'''

gen_imgs_avg = []
gt_imgs_shuffle_avg = []
for i in range (0, 12):
    gen_imgs_avg.append(gen_imgs[:,0,4000*i:4000*(i+1)])
    gt_imgs_shuffle_avg.append(gt_imgs_shuffle[:,0,4000*i:4000*(i+1)])

gen_imgs_avg = np.mean(np.stack(gen_imgs_avg), axis=0)
gt_imgs_shuffle_avg = np.mean(np.stack(gt_imgs_shuffle_avg), axis=0)

print(gen_imgs_avg.shape)
print(gt_imgs_shuffle_avg.shape)

fid_value = eval_fid(gen_imgs_avg, gt_imgs_shuffle_avg)

# print('Evaluation finished')

print('[RSME] MEAN : ' + str(rsme_mean) + ' / VAR : ' + str(rsme_var))
print('[PSNR] MEAN : ' + str(psnr_mean) + ' / VAR : ' + str(psnr_var))
print('[COS] MEAN : ' + str(fre_cos_mean) + ' / VAR : ' + str(fre_cos_var))
print('[MMD] MEAN : ' + str(mmd_mean) + ' / VAR : ' + str(mmd_var))
print('[EMD] MEAN : ' + str(emd_mean) + ' / VAR : ' + str(emd_var))
print('[FID] : ' + str(fid_value / 360.))

gen_imgs_avg = np.swapaxes(gen_imgs[0:4000], 0, 1)
gt_imgs_shuffle_avg = np.swapaxes(gt_imgs_shuffle_avg, 0, 1)

# 데이터 정규화
scaler = StandardScaler()
data_real_scaled = scaler.fit_transform(gt_imgs_shuffle_avg)
data_fake_scaled = scaler.transform(gen_imgs_avg)

# gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
# gen_imgs = 2. * gen_imgs - 1.

combined_data = np.vstack([data_real_scaled, data_fake_scaled])
print(combined_data.shape)
labels = np.array([0]*4000 + [1]*4000)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(combined_data)

# 시각화
plt.figure(figsize=(6, 5))
plt.scatter(
    embedding[labels == 0, 0], embedding[labels == 0, 1],
    c='red', label='Real Data', alpha=0.7
)
plt.scatter(
    embedding[labels == 1, 0], embedding[labels == 1, 1],
    c='blue', label='Synthetic Data', alpha=0.7
)
plt.title("UMAP Projection")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()