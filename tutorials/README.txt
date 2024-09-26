
These tutorials reproduce the figures found in the HippoMaps paper (DeKraker et al. 2024 ). They also serve as examples for common analyses, and to showcase ways of using HippoMaps tools.

Each tutorial is organized into the following sections:
- `config` includes things like input data directory, list of subjects, surface vertex density, etc. This is primarily what a user will edit when running this code on their own data.
- `0)` Mapping of data into surfaces. If `useCheckpoints` is true, this will be skipped and pre-generated numpy arrays will be loaded instead.
- `1)` First step of actual data analysis
- `2+)` Additional steps to break down and explain the analyses being done. These are specific to each tutorial, and include things like postprocessing, plotting, stats, etc.
