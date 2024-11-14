import nibabel as nib
import numpy as np
import hippomaps as hm

print("start test")
map1 = nib.load("/Users/enningyang/Downloads/histology-Blockface_average-4_hemi-mix_den-unfoldiso_label-hipp.shape.gii").darrays[0].data
map2 = nib.load("/Users/enningyang/Downloads/histology-Bieloschowsky_average-4_hemi-mix_den-unfoldiso_label-hipp.shape.gii").darrays[0].data

map_set = np.stack((map1,map2), axis=1)

# record the time
import time
# start = time.time()
# rst1 = hm.stats.contextualize2D(map_set)
# print("time:", time.time()-start)

# #700.595

from stats_new import contextualize2D

start = time.time()
rst2 = contextualize2D(map_set)
print("time:", time.time()-start)
#26.75821805000305

np.allclose(rst1, rst2)
np.save("rst1.npy", rst1)
np.save("rst2.npy", rst2)