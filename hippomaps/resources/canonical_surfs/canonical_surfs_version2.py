import hippomaps.utils
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import shutil


resourcesdir = str(Path(__file__).parents[3]) + "/hippomaps/resources"


unfolded_ref = "/host/bb-compx-01/export02/data/hippunfold_canonical/sub-PNC010/ses-01/surf"
indensity = "unfoldiso"

for outdensity in ["512", "2k", "8k", "18k"]:
    for label in ["hipp", "dentate"]:
        oldcanonical = nib.load(
            f"{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{indensity}_label-{label}_midthickness.surf.gii"
        )
        oldvertices = oldcanonical.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[
            0
        ].data

        insurf = nib.load(
            f"{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{indensity}_label-{label}_midthickness.surf.gii"
        )
        invertices = insurf.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data

        outsurf_name = f"{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{outdensity}_label-{label}_midthickness.surf.gii"
        shutil.copy(f"{unfolded_ref}/sub-PNC010_ses-01_hemi-R_space-unfold_den-{outdensity}_label-{label}_midthickness.surf.gii",outsurf_name)
        outsurf = nib.load(outsurf_name)
        outvertices = outsurf.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[
            0
        ].data
        outfaces = outsurf.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data

        old_x = oldvertices[:, 0]
        old_y = oldvertices[:, 1]
        old_z = oldvertices[:, 2]

        new_x = griddata(
            invertices[:, [1,0]], values=old_x, xi=outvertices[:, :2], method="linear"
        )
        new_y = griddata(
            invertices[:, [1,0]], values=old_y, xi=outvertices[:, :2], method="linear"
        )
        new_z = griddata(
            invertices[:, [1,0]], values=old_z, xi=outvertices[:, :2], method="linear"
        )

        new_x = hippomaps.utils.fillnanvertices(outfaces, new_x)
        new_y = hippomaps.utils.fillnanvertices(outfaces, new_y)
        new_z = hippomaps.utils.fillnanvertices(outfaces, new_z)

        outsurf.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data = np.vstack(
            (new_x, new_y, new_z)
        ).T

        outsurf.to_filename(
            f"{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{outdensity}_label-{label}_midthickness.surf.gii"
        )
