import nibabel as nib
import numpy as np
import hippomaps as hm

print("start test")
map1 = nib.load("/Users/enningyang/Downloads/histology-Blockface_average-4_hemi-mix_den-unfoldiso_label-hipp.shape.gii").darrays[0].data
map2 = nib.load("/Users/enningyang/Downloads/histology-Bieloschowsky_average-4_hemi-mix_den-unfoldiso_label-hipp.shape.gii").darrays[0].data

map_set = np.stack((map1,map2), axis=1)
print(map_set.shape)
hm.stats.contextualize2D(map_set)