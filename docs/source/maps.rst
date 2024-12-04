Map Library 🗺
===================================

All map data is being hosted on the `Open Science Framework <https://osf.io/92p34/>`

Initialization Maps
--------------------
These maps consist of all analyses from the HippoMaps release paper. This includes:
- histology
- ex-vivo 9.4T MRI
- 3T and 7T structural MRI
- 3T and 7T resting-state functional MRI (rsfMRI)
- task-based MRI (Mneumonic similarity task and Episodic memory task)
- iEEG abnd power maps
- morphology (derived with high detail from histology, or stnadard detail from MRI)

Tutorial Checkpoints
-------------
These files are needed to run the tutorials, if `useCheckpoints=True`. Since most of the "heavy lifting" such as preprocessing, sampling data to surfaces, and in the case of task-fMRI, gitting GLMs to timeseries data are already saved in these checkpoints, each tutorial should execute fast!

Note also that tutorials can be easily outfitted to run on new, locally-stored data instead, in case you want to apply similar analyses to your own data!

Share your data
-----------------------
If you have mapped interesting data to hippocampal surfaces, consider sharing it as well! Several projects have already been shared:
- 7T perfusion imaging `paper <https://www.pnas.org/doi/abs/10.1073/pnas.2310044121>`
- 3D Polarized Light Imaging deep extracted features `paper <https://arxiv.org/abs/2402.17744>`

The best way to share your data is to message the `Community <https://hippomaps.readthedocs.io/en/latest/community.html>`

Can I share volumetric data?

Yes! in most cases, it is straightforward to map volumetric data to surfaces by running `HippUnfold <https://hippunfold.readthedocs.io/en/latest/>`. In some cases, you even be able to skip this by mapping directly to pre-generated surfaces in space-MNI152, which can be found `here <https://github.com/khanlab/hippunfold-templateflow>`
