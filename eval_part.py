import numpy as np
import glob
from evaluate.evaluate_utils import rmse,psnr,fre_cosine
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

gen_paths = glob.glob('/data/jionkim/Test/final_result/*.npy')
gt_paths = glob.glob('/data/jionkim/Test/npy_dir_X/*.npy')

gen_imgs = []
gt_imgs = []

for name in gen_paths:
    img = np.load(name)
    gen_imgs.append(img)

for name in gt_paths:
    img = np.load(name)
    gt_imgs.append(img)

gen_imgs = np.array(gen_imgs)
gt_imgs = np.array(gt_imgs)

gen_imgs = gen_imgs[:, :, 0, 0, :]
gen_imgs = np.swapaxes(gen_imgs, 0, 1)
gen_imgs = np.concatenate(gen_imgs, axis=1)

indices = np.random.permutation(gt_imgs.shape[0])
gt_imgs_shuffle = gt_imgs[indices]
gt_imgs_shuffle = gt_imgs_shuffle[:16, :]

gen_imgs = np.expand_dims(gen_imgs, 1)
gt_imgs_shuffle = np.expand_dims(gt_imgs_shuffle, 1)

print(gen_imgs.shape)
print(gt_imgs_shuffle.shape)

rmse_all = rmse(gen_imgs, gt_imgs_shuffle)
psnr_all = psnr(gen_imgs, gt_imgs_shuffle)
fre_cos_all = fre_cosine(gen_imgs, gt_imgs_shuffle)

print('RMSE : ' + str(rmse_all.mean()))
print('PSNR : ' + str(psnr_all.mean().item()))
print('COS : ' + str(fre_cos_all.mean()))

gen_imgs = gen_imgs[:,0,42000:46000]
gt_imgs_shuffle = gt_imgs_shuffle[:,0,42000:46000]

gen_imgs = np.swapaxes(gen_imgs, 0, 1)
gt_imgs_shuffle = np.swapaxes(gt_imgs_shuffle, 0, 1)

# 데이터 정규화
'''
scaler = StandardScaler()
data_real_scaled = scaler.fit_transform(gt_imgs_shuffle)
data_fake_scaled = scaler.transform(gen_imgs)
'''

# gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
# gen_imgs = 2. * gen_imgs - 1.

combined_data = np.vstack([gt_imgs_shuffle, gen_imgs])
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