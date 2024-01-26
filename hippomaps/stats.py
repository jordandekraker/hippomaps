import numpy as np
import nibabel as nib
import scipy.io as spio
from scipy.ndimage import rotate
from scipy.ndimage import shift
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import warnings
import hippomaps.utils
import hippomaps.config
import matplotlib.pyplot as plt


def spin_test(imgfix, imgperm, nperm, metric='pearson', label='hipp', space='orig'):
    """
       Permutation testing of unfolded hippocampus maps.

       Original code by Bradley Karat at https://github.com/Bradley-Karat/Hippo_Spin_Testing
       Karat, B. G., DeKraker, J., Hussain, U., Köhler, S., & Khan, A. R. (2023).
       Mapping the macrostructure and microstructure of the in vivo human hippocampus using diffusion MRI.
       Human Brain Mapping. https://doi.org/10.1002/hbm.26461

       Parameters
       ----------
       imgfix : str
           Path to the fixed map.
       imgperm : str
           Path to the map which will be permuted.
       nperm : int
           Number of permutations to perform.
       metric : str, optional
           Metric for comparing maps (one of pearson, spearman, adjusted rand, or adjusted mutual info).
           Default is 'pearson'. is this true?
       label : str, optional
           Label for the hippocampus. Default is 'hipp'.
       space : str, optional
           Space in which the correlation will be performed.
           If 'orig', the correlation will be performed at the original density.
           If 'unfoldiso', the correlation will be performed at the isotropic density, which is the density used for permutations.
           Default is 'orig'.

       Returns
       -------
       metricnull : Null distribution of the specified metric
       permutedimg : All permuted spatial maps at 'unfoldiso' density.
       r_obs :  The observed association between the two aligned maps.

       pval : p-value based on metricnull r_obs.
       """
    if type(imgfix) == str:
        fixedimg = nib.load(imgfix)
        fixedimgdata = fixedimg.agg_data()
    else:
        fixedimgdata = imgfix
    if type(imgperm) == str:
        permimg = nib.load(imgperm)
        permimgdata = permimg.agg_data()
    else:
        permimgdata = imgperm
    fixedimgvertnum = np.max(fixedimgdata.shape)  # number of vertices
    permimgvertnum = np.max(permimgdata.shape)

    if fixedimgvertnum != permimgvertnum:  # maps don't have to be the same size because they both get interpolated to same density
        warnings.warn("Warning fixed and permuted map not the same size. Program will continue to interpolation")

    vertexnumber = [7262, 2004, 419]  # corresponds to 0p5mm, 1mm, and 2mm respectively
    surfacespacing = ['0p5mm', '1mm', '2mm']

    if fixedimgvertnum not in vertexnumber or permimgvertnum not in vertexnumber:
        raise ValueError(f"Surface number of vertices must be one of {vertexnumber}.")
    else:
        permind = vertexnumber.index(permimgvertnum)
        imgperminterp = hippomaps.utils.density_interp(surfacespacing[permind], 'unfoldiso', permimgdata, label=label,
                                                       method='nearest')[0]
        imgperminterp = np.reshape(imgperminterp, (126, 254))  # get maps to 126x254
        if space == 'unfoldiso':  # if unfoldiso, then need to interpolate fixed image to unfoldiso for correlation
            fixind = vertexnumber.index(
                fixedimgvertnum)  # find the surface spacing which corresponds to the vertex number of that map
            imgfixobs = \
            hippomaps.utils.density_interp(surfacespacing[fixind], 'unfoldiso', fixedimgdata, method='nearest')[
                0]  # interpolate to unfoldiso density
            imgpermobs = imgperminterp.flatten()
            permutedimg = np.empty((126 * 254, nperm))
        elif space == 'orig':  # if orig, then correlations performed at original density, no need to interpolate fixed image
            imgfixobs = fixedimgdata
            imgpermobs = permimgdata
            permutedimg = np.empty((permimgvertnum, nperm))

    rotation = np.random.randint(1, 360, nperm)  # generate random rotations
    translate1 = np.random.randint(-63, 64, nperm)  # generate random translations
    translate2 = np.random.randint(-127, 128, nperm)

    imgsize = imgperminterp.shape
    permutedimgiso = np.empty((imgsize[0], imgsize[1], nperm))
    metricnull = np.empty((nperm))

    for ii in range(nperm):
        rotimg = rotate(imgperminterp, rotation[ii], axes=(1, 0), reshape=False, output=None, order=3, mode='wrap',
                        cval=0.0, prefilter=True)  # rotate image
        transrotimg = shift(rotimg, [translate1[ii], translate2[ii]], output=None, order=3, mode='wrap', cval=0.0,
                            prefilter=True)  # translate image
        permutedimgiso[:, :, ii] = transrotimg  # this is our permuted image at unfoldiso density
        if space == 'orig':  # resample permuted maps back to original density for correlation
            permutedimg[:, ii] = \
            hippomaps.utils.density_interp('unfoldiso', surfacespacing[permind], permutedimgiso[:, :, ii].flatten(),
                                           label=label, method='nearest')[0]
        elif space == 'unfoldiso':  # permuted map can remain in unfoldiso for correlation
            permutedimg[:, ii] = permutedimgiso[:, :, ii].flatten()

    if metric == 'pearson':
        r_obs = pearsonr(imgfixobs, imgpermobs)[0]  # observed correspondance when maps are anatomically aligned
        for ii in range(nperm):
            imgpermflat = permutedimg[:, ii]
            metricnull[ii] = pearsonr(imgfixobs, imgpermflat)[
                0]  # null distribution of correspondance between permuted and fixed map
        r_obs = spearmanr(imgfixobs, imgpermobs)[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:, ii]
            metricnull[ii] = spearmanr(imgfixobs, imgpermflat)[0]
    elif metric == 'adjusted rand':
        r_obs = adjusted_rand_score(imgfixobs, imgpermobs)[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:, ii]
            metricnull[ii] = adjusted_rand_score(imgfixobs, imgpermflat)[0]
    elif metric == 'adjusted mutual info':
        r_obs = adjusted_mutual_info(imgfixobs, imgpermobs)[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:, ii]
            metricnull[ii] = (imgfixobs, imgpermflat)

    pval = np.mean(np.abs(metricnull) >= np.abs(
        r_obs))  # p-value is the sum of all instances where null correspondance is >= observed correspondance / nperm

    return metricnull, permutedimgiso, pval, r_obs


def contextualize2D(taskMaps, n_topComparison=3, permTest=True, nperm=1000, plot2D=True):
    """
       Compares the present maps to thoes in the inital HippoMaps release

       Parameters
       ----------
       taskMaps : a VxT matrix of intensities. T is the number of maps to examine
       n_topComparison : int, optional
           Top N features for which to reurn comparisons (Pearson's R and p)
       plot2D : bool
           Whether to plot the new maps in context of other maps

       Returns
       -------
       topFeatures : Names of most similar intialized features
       topR : Correpsonding Pearson's R values
       topP : Corresponding p values (spin test)
       APcorr : Correlation with AP coordinates
       Subfscorr : Max correlation with subfields (Spearmann)
       ax : axis handle
       """
    nT = taskMaps.shape[1]

    # load required data
    contextHM = np.load('../resources/2Dcontextualize/initialHippoMaps.npz')
    # resample input data if needed
    nVref,iVref = hippomaps.config.get_nVertices(['hipp'],'0p5mm')
    if taskMaps.shape[0] != nVref:
        taskMapsresamp = np.ones((nVref,nT))*np.nan
        for t in range(nT):
            taskMapsresamp[:,t],_,_ = hippomaps.utils.density_interp('2mm','0p5mm',taskMaps[:,t], label='hipp')
    else:
        taskMapsresamp = taskMaps

    # compute correlations with extant features
    topFeatures = np.ones((nT,n_topComparison), dtype='object')
    topR = np.ones((nT,n_topComparison))*np.nan
    topP = np.ones((nT,n_topComparison))*np.nan
    if n_topComparison >0:
        p = np.ones((taskMaps.shape[1],len(contextHM['features'])))
        R = np.ones((taskMaps.shape[1],len(contextHM['features'])))
        for i in range(taskMaps.shape[1]):
            for j in range(len(contextHM['features'])):
                if permTest:
                    _,_,p[i,j],R[i,j] = hippomaps.stats.spin_test(taskMapsresamp[:,i],contextHM['featureData'][:,j], nperm, space='orig')
                else:
                    R[i,j],p[i,j] = pearsonr(taskMapsresamp[:,i],contextHM['featureData'][:,j])
        # get ordering of the closest n_topComparison neighbours
        for t in range(nT):
            order = np.argsort(np.abs(R[t,:]))[::-1]
            for c in range(n_topComparison):
                topFeatures[t,c] = contextHM['features'][order[c]]
                topR[t,c] = R[t,order[c]]
                topP[t,c] = p[t,order[c]]
            
    # get position of new features on 2D space axes
    APcorr = spearmanr(np.concatenate((taskMapsresamp,contextHM['AP'].reshape([-1,1])),axis=1))[0][nT:,:nT]
    APcorr = np.abs(APcorr)
    Subfscorr = spearmanr(np.concatenate((taskMapsresamp,contextHM['subfields_permuted']),axis=1))[0][nT:,:nT]
    Subfscorr = np.nanmax(np.abs(Subfscorr),axis=0)

    # plot in space
    fig, ax = plt.subplots(figsize=(8,8))
    if plot2D:
        ax.spines[['right', 'top']].set_visible(False)
        ax.scatter(contextHM['axiscorrAPPD'][0],contextHM['subfieldsmaxcorr'],c=contextHM['colors'],cmap='Set3',s=200)
        plt.ylabel("absolute subfield correlation (Spearmann's R)")
        plt.xlabel("absolute AP correlation (Pearnson's R)")
        for f,feature in enumerate(contextHM['features']):
            ax.annotate(str(int(contextHM['feature_n'][f])), (contextHM['axiscorrAPPD'][0,f]-.008, contextHM['subfieldsmaxcorr'][f]-.007))
        ax.scatter(APcorr,Subfscorr,color='k',s=200);
        for t in range(nT):
            ax.annotate(str(t), (APcorr[0,t]-.008, Subfscorr[t]-.007),color='w')
            
    return topFeatures, topR, topP, APcorr, Subfscorr, ax
