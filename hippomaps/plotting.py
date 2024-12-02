import numpy as np
import nibabel as nib
import copy
import glob
from scipy.ndimage.filters import gaussian_filter
from numpy.matlib import repmat
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_hemispheres, plot_surf, build_plotter
from brainspace.mesh import mesh_creation as mc
from pathlib import Path

resourcesdir = str(Path(__file__).parents[1]) + '/hippomaps/resources'
import hippomaps.utils


def surfplot_canonical_foldunfold(cdata, hemis=['L', 'R'], labels=['hipp', 'dentate'], unfoldAPrescale=False,
                                  den='0p5mm', tighten_cwindow=False, resourcesdir=resourcesdir, size=[350, 300],
                                  **qwargs):
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

    # load surfaces
    rh = read_surface(
        f'{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{den}_label-hipp_midthickness.surf.gii')
    ru = read_surface(f'{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-hipp_midthickness.surf.gii')
    ru.Points = ru.Points[:, [1, 0, 2]]  # reorient unfolded
    if unfoldAPrescale: ru.Points = area_rescale(ru.Points, den, 'hipp', APaxis=1)
    if len(labels) == 2:
        ud = read_surface(
            f'{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-dentate_midthickness.surf.gii')
        hd = read_surface(
            f'{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{den}_label-dentate_midthickness.surf.gii')
        ud.Points = ud.Points[:, [1, 0, 2]]  # reorient unfolded
        ud.Points = ud.Points + [22, 0, 0]  # translate unfolded dg
        if unfoldAPrescale: ud.Points = area_rescale(ud.Points, den, 'dentate', APaxis=1)
        # add to original
        npts = rh.n_points
        rh = mc.build_polydata(np.concatenate((rh.Points.copy(), hd.Points.copy())),
                               cells=np.concatenate((rh.GetCells2D().copy(), hd.GetCells2D().copy() + npts)))
        ru = mc.build_polydata(np.concatenate((ru.Points.copy(), ud.Points.copy())),
                               cells=np.concatenate((ru.GetCells2D().copy(), ud.GetCells2D().copy() + npts)))

    # flip to get left hemisphere
    lh = mc.build_polydata(rh.Points.copy(), cells=rh.GetCells2D().copy())
    lh.Points[:, 0] = -lh.Points[:, 0]
    lu = mc.build_polydata(ru.Points.copy(), cells=ru.GetCells2D().copy())
    lu.Points[:, 0] = -lu.Points[:, 0]

    # do some cdata formatting
    cdata = np.reshape(cdata, [cdata.shape[0], len(hemis), -1])
    if len(cdata.shape) == 2: cdata = np.expand_dims(cdata, axis=2)
    if tighten_cwindow > 0:
        for i in range(0, cdata.shape[2]):
            cdata[:, :, i] = bound_cdata(cdata[:, :, i])

    # set up layout
    surfDict = {'Lf': lh, 'Lu': lu, 'Rf': rh, 'Ru': ru}
    surfList = np.ones((cdata.shape[2], len(hemis) * 2), dtype=object)
    arrName = np.ones((cdata.shape[2], len(hemis) * 2), dtype=object)
    for h, hemi in enumerate(hemis):
        if hemi == 'L':
            surfList[:, [h, h + 1]] = np.array([f"{hemi}f", f"{hemi}u"])
            for f in range(cdata.shape[2]):
                lh.append_array(cdata[:, h, f], name=f'feature{f}', at='point')
                lu.append_array(cdata[:, h, f], name=f'feature{f}', at='point')
        elif hemi == 'R':
            surfList[:, [h * 2, h * 2 + 1]] = np.array([f"{hemi}u", f"{hemi}f"])
            for f in range(cdata.shape[2]):
                rh.append_array(cdata[:, h, f], name=f'feature{f}', at='point')
                ru.append_array(cdata[:, h, f], name=f'feature{f}', at='point')
        for f in range(cdata.shape[2]):
            arrName[f, :] = f'feature{f}'

    # extra parameters
    new_qwargs = dict()
    if not 'zoom' in qwargs:
        new_qwargs['zoom'] = 1.7
    if not 'nan_color' in qwargs:
        new_qwargs['nan_color'] = (0, 0, 0, 0)
    new_qwargs.update(qwargs)
    new_size = copy.deepcopy(size)
    new_size[0] = new_size[0] * len(hemis)
    new_size[1] = new_size[1] * cdata.shape[2]
    if 'color_bar' in qwargs:
        new_size[0] = new_size[0] + 60
    p = plot_surf(surfDict, surfList, array_name=arrName, size=new_size, **new_qwargs)
    return p


def surfplot_sub_foldunfold(hippunfold_dir, sub, ses, features, hemis=['L', 'R'], labels=['hipp', 'dentate'],
                            flipRcurv=True, unfoldAPrescale=False, den='0p5mm', modality='T1w', tighten_cwindow=True,
                            rotate=20, resourcesdir=resourcesdir, size=[350, 230], cmap='viridis', **qwargs):
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
            Whether to flip the curvature map for the right hemisphere. Default is True.
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

    if len(ses) > 0:
        ses = 'ses-' + ses
        uses = '_' + ses
    else:
        uses = ''

    # load surfaces
    surf = []
    n = 0
    for hemi in hemis:
        for space in [modality, 'unfold']:
            for label in labels:
                fn1 = f'{hippunfold_dir}/sub-{sub}/{ses}/surf/sub-{sub}{uses}_hemi-{hemi}_space-{space}_den-{den}_label-{label}_midthickness.surf.gii'
                if glob.glob(fn1):
                    s = read_surface(fn1)
                else:
                    print(fn1 + ' not found, using generic')
                    s = read_surface(
                        f'{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-{label}_midthickness.surf.gii')
                    if space != 'unfold': s.Points = np.ones(s.Points.shape) * np.nan
                if label == 'hipp':
                    nptsHipp = s.n_points
                if space == 'unfold':
                    s.Points = s.Points[:, [1, 0, 2]]  # reorient unfold
                    if unfoldAPrescale: s.Points = area_rescale(s.Points, den, label, APaxis=1)
                    if label == 'dentate':
                        s.Points = s.Points + [22, 0, 0]  # translate DG
                if hemi == "L" and space == 'unfold':  # flip L unfolded
                    s.Points[:, 0] = -s.Points[:, 0]
                if label == 'dentate':  # concatenate dentate to hipp (hipp should be immediately prior)
                    s = mc.build_polydata(np.concatenate((oldsurf.Points.copy(), s.Points.copy())),
                                          cells=np.concatenate(
                                              (oldsurf.GetCells2D().copy(), s.GetCells2D().copy() + nptsHipp)))
                oldsurf = s

            surf.append(s)
    npts = len(surf[0].Points)

    # rotate surfaces to approx coronal-oblique
    if rotate:
        #aff = np.loadtxt(f'{resourcesdir}/xfms/corobl-20deg_xfm.txt')
        aff = np.eye(4)
        aff[1,1] = np.cos(np.deg2rad(rotate))
        aff[2,2] = np.cos(np.deg2rad(rotate))
        aff[1,2] = np.sin(np.deg2rad(rotate))
        aff[2,1] = -np.sin(np.deg2rad(rotate))
        for i, h in enumerate(hemis):
            p = np.hstack((surf[i * 2].Points, np.ones((npts, 1))))
            p = p @ aff
            surf[i * 2].Points = p[:, :3]

    # load features data
    ind = [range(nptsHipp), range(nptsHipp, npts)]
    cdata = np.ones([npts, len(hemis), len(features)]) * np.nan
    for h, hemi in enumerate(hemis):
        for l, label in enumerate(labels):
            for f, feature in enumerate(features):
                if feature == 'subfields':
                    type = 'label'
                else:
                    type = 'shape'
                fn2 = f'{hippunfold_dir}/sub-{sub}/{ses}/surf/sub-{sub}{uses}_hemi-{hemi}_space-{modality}_den-{den}_label-{label}_*{feature}*.{type}.gii'
                fn3 = glob.glob(fn2)
                try:
                    cdata[ind[l], h, f] = nib.load(fn3[0]).darrays[0].data
                    if flipRcurv and feature == 'curvature' and hemi == 'L':
                        cdata[ind[l], h, f] = -cdata[ind[l], h, f]
                except:
                    print(fn2 + ' failed')

    # options
    cmaps = np.ones((len(features), len(hemis) * 2), dtype=object)
    cmaps[:] = cmap
    for f, feature in enumerate(features):
        if feature == 'subfields':
            cdata[ind[1], :, f] = np.nanmax(cdata[ind[0], :, f]) + 1
            cmaps[f, :] = ('jet')
        else:
            if tighten_cwindow > 0:
                cdata[:, :, f] = bound_cdata(cdata[:, :, f])

    # set up layout
    surfDict = {}
    surfList = np.ones((len(features), len(hemis) * 2), dtype=object)
    arrName = np.ones((len(features), len(hemis) * 2), dtype=object)
    for h, hemi in enumerate(hemis):
        surfDict.update({f"{hemi}f": surf[h * 2]})
        surfDict.update({f"{hemi}u": surf[h * 2 + 1]})
        if hemi == 'L':
            surfList[:, [h, h + 1]] = np.array([f"{hemi}f", f"{hemi}u"])
        elif hemi == 'R':
            surfList[:, [h * 2, h * 2 + 1]] = np.array([f"{hemi}u", f"{hemi}f"])
        for f, feature in enumerate(features):
            surf[h * 2].append_array(cdata[:, h, f], name=feature, at='point')
            surf[h * 2 + 1].append_array(cdata[:, h, f], name=feature, at='point')
            arrName[f, :] = feature

    # extra parameters
    new_qwargs = dict()
    if not 'zoom' in qwargs:
        new_qwargs['zoom'] = 1.5
    if not 'nan_color' in qwargs:
        new_qwargs['nan_color'] = (0, 0, 0, 0)
    new_qwargs.update(qwargs)
    new_size = copy.deepcopy(size)
    new_size[0] = new_size[0] * len(hemis)
    new_size[1] = new_size[1] * cdata.shape[2]
    if 'color_bar' in qwargs:
        new_size[0] = new_size[0] + 70

    p = plot_surf(surfDict, surfList, array_name=arrName, size=new_size, cmap=cmaps, **new_qwargs)
    return p


def bound_cdata(cdata, cutoff=0.05):
    """
      Returns upper and lower X percent interval values.
      Parameters
      ----------
      cdata :  List of values.
      cutoff : Default is 0.05.
      Returns
      -------
       Values within the upper and lower X percent interval.
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
    except:
        print('cdata all NaN')
    return np.reshape(cdata, shp)


def area_rescale(vertices, den, label, APaxis=1):
    """
         Rescales unfolded surface vertices to account for overrepresentation of anterior and posterior regions.

         Parameters
         ----------
         vertices : Unfolded surface vertices.
         den : str
             Density of the unfolded space. One of '0p5mm', '1mm', '2mm', or 'unfoldiso'.
         label : str
             Surface label. Either 'hipp' or 'dentate'.
         APaxis : int, optional
             Axis along the anterior-posterior direction. Default is 1.
         Returns
         -------
         numpy.ndarray
             Rescaled unfolded surface vertices.
     """
    w = 126 if label == 'hipp' else 30  # width of unfolded space
    s = 15  # surf area smoothing (sigma)
    Pold = vertices[:, APaxis]

    # compute rescaling from surface areas
    surfarea = \
        nib.load(
            f'{resourcesdir}/canonical_surfs/tpl-avg_space-unfold_den-{den}_label-{label}_surfarea.shape.gii').darrays[
            0].data
    surfarea = hippomaps.utils.density_interp(den, 'unfoldiso', surfarea.flatten(), label)
    surfarea = np.reshape(surfarea, (w, 254))
    surfarea = gaussian_filter(surfarea, sigma=s)

    avg_surfarea = np.mean(surfarea, axis=0)
    rescalefactor = np.cumsum(1 / avg_surfarea)
    rescalefactor = rescalefactor - np.min(rescalefactor)
    rescalefactor = rescalefactor / np.max(rescalefactor)
    rescalefactor = rescalefactor + 1 - np.linspace(0, 1, len(rescalefactor))

    rescalefactor = repmat(rescalefactor, w, 1)
    rescalefactor = hippomaps.utils.density_interp('unfoldiso', den, rescalefactor.flatten(), label)

    Pnew = Pold * rescalefactor
    vertices[:, APaxis] = Pnew
    return vertices
