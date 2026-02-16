import numpy as np
import nibabel as nib
import copy
import glob
import warnings
from scipy.ndimage.filters import gaussian_filter
from numpy.matlib import repmat
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_hemispheres, plot_surf, build_plotter
from brainspace.mesh import mesh_creation as mc
from pathlib import Path

resourcesdir = str(Path(__file__).parents[1]) + '/hippomaps/resources'
import hippomaps.utils


# ----------------------------
# Small, reusable helpers
# (kept compact, but module-wide)
# ----------------------------

_V2_DENS = {"2k", "8k", "18k"}


def _is_v2_den(den: str) -> bool:
    return den in _V2_DENS


def _ses_dir_and_suffix(ses: str):
    """Return (ses_dir, uses_suffix) where ses_dir is 'ses-XX' or '' and uses_suffix is '_ses-XX' or ''."""
    if ses:
        ses_dir = f"ses-{ses}" if not ses.startswith("ses-") else ses
        uses = f"_{ses_dir}"
        return ses_dir, uses
    return "", ""


def _subj_root(hippunfold_dir, sub: str, ses_dir: str) -> Path:
    root = Path(hippunfold_dir) / f"sub-{sub}"
    return (root / ses_dir) if ses_dir else root


def _canonical_surf(resourcesdir, den: str, label: str, space: str) -> str:
    return str(Path(resourcesdir) / "canonical_surfs" / f"tpl-avg_space-{space}_den-{den}_label-{label}_midthickness.surf.gii")


def _subject_surf(subjroot: Path, sub: str, uses: str, hemi: str, space: str, den: str, label: str) -> str:
    return str(subjroot / "surf" / f"sub-{sub}{uses}_hemi-{hemi}_space-{space}_den-{den}_label-{label}_midthickness.surf.gii")


def _metric_glob(subjroot: Path, sub: str, uses: str, hemi: str, den: str, label: str,
                 feature: str, modality: str, is_v2: bool) -> str:
    """Return a glob pattern for metric/shape files that works for v1 and v2 conventions."""
    gii_type = "label" if feature == "subfields" else "shape"
    # v1 includes space-{modality}; v2 does not
    space_part = f"_space-{modality}" if not is_v2 else ""
    # IMPORTANT: avoid introducing double underscores (the bug in the original)
    return str(subjroot / ("metric" if is_v2 else "surf") / f"sub-{sub}{uses}_hemi-{hemi}{space_part}_den-{den}_label-{label}_*{feature}*.{gii_type}.gii")


def _apply_unfold_transforms(s, den: str, label: str, unfoldAPrescale: bool, is_v2: bool, hemi: str):
    # v1 needed a reorient; v2 unfolded surfaces are already oriented (PR #534), so skip swap for v2
    if not is_v2:
        s.Points = s.Points[:, [1, 0, 2]]  # reorient unfold (v1)

    if unfoldAPrescale:
        s.Points = area_rescale(s.Points, den, label, APaxis=1)

    if label == "dentate":
        # DG should be shifted outward from hippocampus; with hemi-aware unfold space,
        # the direction differs by hemisphere.
        dg_shift = 22.0 if hemi == "R" else -22.0
        s.Points = s.Points + [dg_shift, 0, 0]

    return s


def _concat_hipp_dentate(hipp_surf, dent_surf, nptsHipp: int):
    return mc.build_polydata(
        np.concatenate((hipp_surf.Points.copy(), dent_surf.Points.copy())),
        cells=np.concatenate((hipp_surf.GetCells2D().copy(),
                              dent_surf.GetCells2D().copy() + nptsHipp))
    )


def _rotate_in_place(polydata, rotate_deg: float):
    """Rotate points around x-axis (coronal-oblique-ish) using a 4x4 affine."""
    if not rotate_deg:
        return
    aff = np.eye(4)
    c = np.cos(np.deg2rad(rotate_deg))
    s = np.sin(np.deg2rad(rotate_deg))
    aff[1, 1] = c
    aff[2, 2] = c
    aff[1, 2] = s
    aff[2, 1] = -s
    pts = polydata.Points
    p = np.hstack((pts, np.ones((pts.shape[0], 1))))
    polydata.Points = (p @ aff)[:, :3]


def _default_plot_kwargs(qwargs, zoom=1.5):
    out = {}
    if "zoom" not in qwargs:
        out["zoom"] = zoom
    if "nan_color" not in qwargs:
        out["nan_color"] = (0, 0, 0, 0)
    out.update(qwargs)
    return out


def _scaled_size(size, nhemis: int, nrows: int, colorbar_pad: int):
    new_size = list(copy.deepcopy(size))
    new_size[0] = int(new_size[0] * nhemis)
    new_size[1] = int(new_size[1] * nrows)
    return new_size, (new_size[0] + colorbar_pad, new_size[1])


# ----------------------------
# Public API
# ----------------------------

def surfplot_canonical_foldunfold(
    cdata, hemis=['L', 'R'], labels=['hipp', 'dentate'],
    unfoldAPrescale=False, den='0p5mm', tighten_cwindow=False,
    resourcesdir=resourcesdir, size=[350, 300], **qwargs
):
    """
       Plots canonical folded and unfolded surfaces for hippocampus and dentate gyrus.
       Parameters
       ----------
       cdata : numpy.ndarray
           Array with the shape Vx2xF, where V is the number of vertices (including DG unless specified),
           2 is the number of hemispheres (unless specified), and F is the number of rows/features.
       hemis : list of str, optional
           List of hemispheres to visualize. Default is ['L', 'R'].
       labels : list of str, optional
           List of labels for different structures. Default is ['hipp', 'dentate'].
       unfoldAPrescale : bool, optional
           Whether to pre-scale the anterior-posterior axis during unfolding. Default is False.
       den : str, optional
           Density parameter for surface plot. Default is '0p5mm'.
       tighten_cwindow : bool, optional
           Whether to tighten the color window for the surface plot. Default is False.
       resourcesdir : str, optional
           Directory path containing additional resources. Default is the value of resourcesdir.
       size : list of int, optional
           Size of the surface plot. Default is [350, 300].
       **qwargs : dict, optional
           Additional keyword arguments for customization.
           See https://brainspace.readthedocs.io/en/latest/generated/brainspace.plotting.surface_plotting.plot_surf.html#brainspace.plotting.surface_plotting.plot_surf

       Returns
       -------
       matplotlib.figure.Figure
           Figure object for the generated surface plot.
       Notes
       -----
       This function is suitable for plotting canonical folded and unfolded surfaces, and it is particularly useful
       when the data (`cdata`) isn't specific to one subject (e.g., maybe it has been averaged across many subjects).
    """

    is_v2 = _is_v2_den(den)
    if is_v2 and unfoldAPrescale:
        warnings.warn("Hippunfold v2 outputs do not require unfoldAPrescale")
        unfoldAPrescale = False

    # load canonical surfaces (right hemi as template)
    rh = read_surface(_canonical_surf(resourcesdir, den, "hipp", "canonical"))
    ru = read_surface(_canonical_surf(resourcesdir, den, "hipp", "unfold"))
    ru.Points = ru.Points[:, [1, 0, 2]]  # reorient unfolded
    if unfoldAPrescale:
        ru.Points = area_rescale(ru.Points, den, "hipp", APaxis=1)

    if len(labels) == 2:
        ud = read_surface(_canonical_surf(resourcesdir, den, "dentate", "unfold"))
        hd = read_surface(_canonical_surf(resourcesdir, den, "dentate", "canonical"))
        ud.Points = ud.Points[:, [1, 0, 2]]
        ud.Points = ud.Points + [22, 0, 0]
        if unfoldAPrescale:
            ud.Points = area_rescale(ud.Points, den, "dentate", APaxis=1)

        npts = rh.n_points
        rh = _concat_hipp_dentate(rh, hd, npts)
        ru = _concat_hipp_dentate(ru, ud, npts)

    # left hemi is x-flipped copy
    lh = mc.build_polydata(rh.Points.copy(), cells=rh.GetCells2D().copy())
    lh.Points[:, 0] = -lh.Points[:, 0]
    lu = mc.build_polydata(ru.Points.copy(), cells=ru.GetCells2D().copy())
    lu.Points[:, 0] = -lu.Points[:, 0]

    # format cdata to (V, H, F)
    cdata = np.reshape(cdata, [cdata.shape[0], len(hemis), -1])
    if len(cdata.shape) == 2:
        cdata = np.expand_dims(cdata, axis=2)

    if tighten_cwindow:
        for i in range(cdata.shape[2]):
            cdata[:, :, i] = bound_cdata(cdata[:, :, i])

    # layout
    surfDict = {"Lf": lh, "Lu": lu, "Rf": rh, "Ru": ru}
    surfList = np.ones((cdata.shape[2], len(hemis) * 2), dtype=object)
    arrName = np.ones((cdata.shape[2], len(hemis) * 2), dtype=object)

    for h, hemi in enumerate(hemis):
        if hemi == "L":
            surfList[:, [h, h + 1]] = np.array([f"{hemi}f", f"{hemi}u"])
            folded, unfolded = lh, lu
        else:  # "R"
            surfList[:, [h * 2, h * 2 + 1]] = np.array([f"{hemi}u", f"{hemi}f"])
            folded, unfolded = rh, ru

        for f in range(cdata.shape[2]):
            name = f"feature{f}"
            folded.append_array(cdata[:, h, f], name=name, at="point")
            unfolded.append_array(cdata[:, h, f], name=name, at="point")
            arrName[f, :] = name

    new_qwargs = _default_plot_kwargs(qwargs, zoom=1.7)

    new_size = copy.deepcopy(size)
    new_size[0] *= len(hemis)
    new_size[1] *= cdata.shape[2]
    if "color_bar" in qwargs:
        new_size[0] += 60

    return plot_surf(surfDict, surfList, array_name=arrName, size=new_size, **new_qwargs)


def surfplot_sub_foldunfold(
    hippunfold_dir, sub, ses, features,
    hemis=['L', 'R'], labels=['hipp', 'dentate'],
    flipRcurv=True, unfoldAPrescale=False,
    den='0p5mm', modality='T1w',
    tighten_cwindow=True, rotate=20,
    resourcesdir=resourcesdir, size=[350, 230],
    cmap='viridis', **qwargs
):
    """
        Plots subject-specific folded and unfolded surfaces (hipp/dentate; folded/unfolded).

        Parameters
        ----------
        hippunfold_dir: str
            Directory path containing unfolded hippocampus data.
        sub: str
            Subject ID. Inputs are path/filenames
        ses: str
            Session ID. Inputs are path/filenames
        features: str
            Feature or measurement to visualize on the surface plot. Can include thickness, curvature, gyrification, subfields,
            or other added data that follows the same naming convention.
        hemis: list of str, optional
            List of hemispheres to visualize. Default is ['L', 'R'].
        labels: list of str, optional
            List of labels for different structures. Default is ['hipp', 'dentate'].
        flipRcurv: bool, optional
            Whether to flip the curvature map for the right hemisphere. Default is True for _v1 but False for _v2.
        unfoldAPrescale: bool, optional
            Whether to pre-scale the anterior-posterior axis during unfolding. Default is False.
        den: str, optional
            Density parameter for surface plot. Default is '0p5mm'.
        modality: str, optional
            Imaging modality (e.g., 'T1w'). Default is 'T1w'.
        tighten_cwindow: bool, optional
            Whether to tighten the color window for the surface plot. Default is True.
        rotate: bool, optional
            Whether to rotate the surface plot. Default is True.
        resourcesdir: str, optional
            Directory path containing additional resources. Default is the value of resourcesdir.
        size: list of int, optional
            Size of the surface plot. Default is [350, 230].
        cmap: str, optional
            Colormap for the surface plot. Default is 'viridis'.
        **qwargs: dict, optional
            Additional keyword arguments for customization.

        Returns
        -------
        matplotlib.figure.Figure
            The function generates a surface plot for the specified subject's folded and unfolded hippocampus.
    """

    is_v2 = _is_v2_den(den)
    if is_v2:
        if unfoldAPrescale:
            warnings.warn("Hippunfold v2 outputs do not require unfoldAPrescale")
            unfoldAPrescale = False
        flipRcurv = False

    ses_dir, uses = _ses_dir_and_suffix(ses)
    subjroot = _subj_root(hippunfold_dir, sub, ses_dir)

    # -----------------------
    # load surfaces (and optionally concat DG)
    # order in `surf` is per hemi: [folded(concat), unfolded(concat)]
    # -----------------------
    surf = []
    nptsHipp = None

    for hemi in hemis:
        for space in [modality, "unfold"]:
            oldsurf = None
            for label in labels:
                fn = _subject_surf(subjroot, sub, uses, hemi, space, den, label)
                if glob.glob(fn):
                    s = read_surface(fn)
                else:
                    print(fn + " not found, using generic")
                    s = read_surface(_canonical_surf(resourcesdir, den, label, "unfold"))
                    if space != "unfold":
                        s.Points = np.full_like(s.Points, np.nan)

                if label == "hipp":
                    nptsHipp = s.n_points

                if space == "unfold":
                    s = _apply_unfold_transforms(s, den, label, unfoldAPrescale, is_v2=is_v2, hemi=hemi)

                if label == "dentate":
                    if oldsurf is None or nptsHipp is None:
                        raise RuntimeError("Dentate encountered before hippocampus; labels order must be ('hipp','dentate').")
                    s = _concat_hipp_dentate(oldsurf, s, nptsHipp)

                oldsurf = s

            surf.append(s)

    npts = len(surf[0].Points)

    # rotate folded surfaces only (index 0,2,...)
    if rotate:
        for i in range(len(hemis)):
            _rotate_in_place(surf[i * 2], rotate)

    # -----------------------
    # load features
    # -----------------------
    ind = [range(nptsHipp), range(nptsHipp, npts)]
    cdata = np.full((npts, len(hemis), len(features)), np.nan)

    for h, hemi in enumerate(hemis):
        for l, label in enumerate(labels):
            for f, feature in enumerate(features):
                pattern = _metric_glob(subjroot, sub, uses, hemi, den, label, feature, modality, is_v2)
                matches = glob.glob(pattern)
                if not matches:
                    print(pattern + " failed (no matches)")
                    continue
                try:
                    cdata[ind[l], h, f] = nib.load(matches[0]).darrays[0].data
                    # FIXED: flip curvature for RIGHT hemisphere (not L)
                    if flipRcurv and feature == "curvature" and hemi == "R":
                        cdata[ind[l], h, f] = -cdata[ind[l], h, f]
                except Exception as e:
                    print(matches[0] + f" failed ({e})")

    # -----------------------
    # display options
    # -----------------------
    cmaps = np.ones((len(features), len(hemis) * 2), dtype=object)
    cmaps[:] = cmap

    for f, feature in enumerate(features):
        if feature == "subfields":
            cdata[ind[1], :, f] = np.nanmax(cdata[ind[0], :, f]) + 1
            cmaps[f, :] = "jet"
        elif tighten_cwindow:
            cdata[:, :, f] = bound_cdata(cdata[:, :, f])

    # -----------------------
    # layout + attach arrays
    # -----------------------
    surfDict = {}
    surfList = np.ones((len(features), len(hemis) * 2), dtype=object)
    arrName = np.ones((len(features), len(hemis) * 2), dtype=object)

    for h, hemi in enumerate(hemis):
        surfDict[f"{hemi}f"] = surf[h * 2]
        surfDict[f"{hemi}u"] = surf[h * 2 + 1]

        # preserve your original ordering behavior
        if hemi == "L":
            surfList[:, [h, h + 1]] = np.array([f"{hemi}f", f"{hemi}u"])
        else:  # "R"
            surfList[:, [h * 2, h * 2 + 1]] = np.array([f"{hemi}u", f"{hemi}f"])

        for f, feature in enumerate(features):
            surf[h * 2].append_array(cdata[:, h, f], name=feature, at="point")
            surf[h * 2 + 1].append_array(cdata[:, h, f], name=feature, at="point")
            arrName[f, :] = feature

    new_qwargs = _default_plot_kwargs(qwargs, zoom=1.5)

    new_size = copy.deepcopy(size)
    new_size[0] *= len(hemis)
    new_size[1] *= cdata.shape[2]
    if "color_bar" in qwargs:
        new_size[0] += 70

    return plot_surf(surfDict, surfList, array_name=arrName, size=new_size, cmap=cmaps, **new_qwargs)


def bound_cdata(cdata, cutoff=0.05):
    """
    Returns upper and lower X percent interval values.
    """
    if not cutoff:
        return False
    shp = cdata.shape
    c = cdata.flatten()
    l = np.sort(c[~np.isnan(c)])
    try:
        bounds = l[[int(cutoff * len(l)), int((1 - cutoff) * len(l))]]
        cdata[cdata < bounds[0]] = bounds[0]
        cdata[cdata > bounds[1]] = bounds[1]
    except Exception:
        print("cdata all NaN")
    return np.reshape(cdata, shp)


def area_rescale(vertices, den, label, APaxis=1):
    """
    Rescales unfolded surface vertices to account for overrepresentation of anterior and posterior regions.
    """
    w = 126 if label == "hipp" else 30  # width of unfolded space
    s = 15  # surf area smoothing (sigma)
    Pold = vertices[:, APaxis]

    surfarea = nib.load(
        f"{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-{label}_surfarea.shape.gii"
    ).darrays[0].data
    surfarea = hippomaps.utils.density_interp(den, "unfoldiso", surfarea.flatten(), label)
    surfarea = np.reshape(surfarea, (w, 254))
    surfarea = gaussian_filter(surfarea, sigma=s)

    avg_surfarea = np.mean(surfarea, axis=0)
    rescalefactor = np.cumsum(1 / avg_surfarea)
    rescalefactor = rescalefactor - np.min(rescalefactor)
    rescalefactor = rescalefactor / np.max(rescalefactor)
    rescalefactor = rescalefactor + 1 - np.linspace(0, 1, len(rescalefactor))

    rescalefactor = repmat(rescalefactor, w, 1)
    rescalefactor = hippomaps.utils.density_interp("unfoldiso", den, rescalefactor.flatten(), label)

    vertices[:, APaxis] = Pold * rescalefactor
    return vertices
