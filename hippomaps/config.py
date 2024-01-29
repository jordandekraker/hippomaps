import numpy as np
import nibabel as nib
from pathlib import Path
resourcesdir = str(Path(__file__).parents[1]) + '/resources'

labelNames = ['hipp','dentate']
densityNames = ['2mm','1mm','0p5mm','unfoldiso']

nVertices = np.zeros((len(labelNames), len(densityNames))).astype(int)
for l,label in enumerate(labelNames):
    for d,den in enumerate(densityNames):
        surf = nib.load(f'{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-{label}_midthickness.surf.gii')
        nVertices[l,d] = int(surf.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data.shape[0])

def get_nVertices(labels,den):
    iLabels = np.where(np.in1d(labelNames, labels))[0]
    idens = np.where(np.in1d(densityNames, den))[0]
    nV = np.sum(nVertices[iLabels,idens])
    iV = []
    iV.append(range(nVertices[iLabels[0],idens[0]]))
    if len(iLabels)>1:
       iV.append(range(nVertices[iLabels[0],idens[0]], nV))
    return nV,iV
