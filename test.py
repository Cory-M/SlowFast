import slowfast.datasets.transform as ct
import torch
import pdb

imgs = torch.randn(3,4, 5, 5) #CTHW
img = imgs[:,0,:,:].clone()
#new_img = ct._hsv2rgb(ct._rgb2hsv(img)) 
new_img = ct.hue_jitter(0, img)
new_imgs = ct.hue_jitter(0.1, imgs)
#new_imgs = ct._hsv2rgb(ct._rgb2hsv(imgs))
print(new_imgs[:,0,:,:] == new_img)
pdb.set_trace()

#new = []
#for x in torch.unbind:
#	new.append(ct.hue_jitter(0.1, x))
#
#pdb.set_trace()

