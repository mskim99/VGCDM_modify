import numpy as np
from evaluate.evaluate_utils import rmse,psnr,fre_cosine

img1 = np.load('/data/jionkim/signal_dataset/final_result_concat.npy')
img2 = np.load('/data/jionkim/signal_dataset/G1EpoxyRasterPlate_Movement_X_train1.npy')

img1 = img1.reshape(1, 1, -1)
img2 = img2.reshape(1, 1, -1)

rmse_all = rmse(img1, img2)
psnr_all = psnr(img1, img2)
fre_cos_all = fre_cosine(img1, img2)

print(rmse_all)
print(psnr_all)
print(fre_cos_all)