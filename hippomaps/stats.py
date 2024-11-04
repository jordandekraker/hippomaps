import numpy as np
import nibabel as nib
import pandas as pd
from tabulate import tabulate
from scipy.ndimage import rotate, shift
from scipy.stats import spearmanr, pearsonr
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import warnings
import hippomaps.utils
import hippomaps.config
import matplotlib.pyplot as plt
from brainspace.mesh.mesh_io import read_surface
from hippomaps.moran import MoranRandomization
from brainspace.mesh import mesh_elements as me
from eigenstrapping import SurfaceEigenstrapping
from pathlib import Path
resourcesdir=str(Path(__file__).parents[1]) + '/resources'


def spin_test(imgfix, imgperm, nperm=1000, metric='pearsonr', label='hipp', den='0p5mm'):
    """
       Permutation testing of unfolded hippocampus maps.

       Original code by Bradley Karat at https://github.com/Bradley-Karat/Hippo_Spin_Testing
       Karat, B. G., DeKraker, J., Hussain, U., Köhler, S., & Khan, A. R. (2023).
       Mapping the macrostructure and microstructure of the in vivo human hippocampus using diffusion MRI.
       Human Brain Mapping. https://doi.org/10.1002/hbm.26461

       Parameters
       ----------
       imgfix : str or array
           Path to the fixed map, or fixed array map.
       imgperm : str or array
           Path to the fixed map to be permuted, or array map to be permuted.
       nperm : int
           Number of permutations to perform.
       metric : str, optional
           Metric for comparing maps (one of pearsonr, spearmanr, adjusted_mutual_info_score, or adjusted_mutual_info_score).
           Default is 'pearson'.
       label : str, optional
           Label for the hippocampus ('hipp' or 'dentate'). Default is 'hipp'.
       den : str, optional
           Density of the surface data. Default '0p5mm'.

       Returns
       -------
       metricnull : Null distribution of the specified metric
       permutedimg : All permuted spatial maps at 'unfoldiso' density.
       r_obs :  The observed association between the two aligned maps.
       pval : p-value based on metricnull r_obs.
       """
    if type(imgfix) == str:
        imgfix = nib.load(imgfix).agg_data()
    if type(imgperm) == str:
        imgperm = nib.load(imgperm).agg_data()

    # resmaple to space-unfoldiso
    if den != 'unfoldiso':
        imgperm = hippomaps.utils.density_interp(den, 'unfoldiso', imgperm, label=label, method='nearest')[0]
        imgfix = hippomaps.utils.density_interp(den, 'unfoldiso', imgfix, label=label, method='nearest')[0]  
    if label == 'hipp':
        imgperm = np.reshape(imgperm, (126, 254))  # get maps to 126x254
        imgfix = np.reshape(imgfix, (126, 254))  # get maps to 126x254
    elif label == 'dentate':
        imgperm = np.reshape(imgperm, (32, 254)) # get maps to 32x254
        imgfix = np.reshape(imgfix, (32, 254))

    rotation = np.random.randint(1, 360, nperm)  # generate random rotations
    translate1 = np.random.randint(-63, 64, nperm)  # generate random translations
    translate2 = np.random.randint(-127, 128, nperm)
    permutedimg = np.empty((126, 254, nperm))
    metricnull = np.empty((nperm))

    for ii in range(nperm):
        rotimg = rotate(imgperm, rotation[ii], axes=(1, 0), reshape=False, output=None, order=3, mode='wrap',
                        cval=0.0, prefilter=True)  # rotate image
        transrotimg = shift(rotimg, [translate1[ii], translate2[ii]], output=None, order=3, mode='wrap', cval=0.0,
                            prefilter=True)  # translate image
        permutedimg[:, :, ii] = transrotimg  # this is our permuted image at unfoldiso density

    r_obs = pearsonr(imgfix.flatten(), imgperm.flatten())[0]  
    for ii in range(nperm):
        metricnull[ii] = eval(metric)(imgfix.flatten(), permutedimg[:,:,ii].flatten())[0]  
    
    # p-value is the sum of all instances where null correspondance is >= observed correspondance / nperm
    pval = np.mean(np.abs(metricnull) >= np.abs(r_obs))  

    return metricnull, permutedimg, pval, r_obs


def moran_test(imgfix, imgperm, nperm=1000, metric='pearsonr', label='hipp', den='0p5mm'):
    """
       Moran Spectral Randomization
       Moran Spectral Randomization (MSR) computes Moran’s I, a metric for spatial auto-correlation and generates normally distributed data with similar auto-correlation. MSR relies on a weight matrix denoting the spatial proximity of features to one another. Within neuroimaging, one straightforward example of this is inverse geodesic distance i.e. distance along the cortical surface.

       Code from BrainSpace https://brainspace.readthedocs.io/en/latest/python_doc/auto_examples/plot_tutorial3.html
       Vos de Wael, Reinder, Oualid Benkarim, Casey Paquola, Sara Lariviere, Jessica Royer, Shahin Tavakol, Ting Xu et al. "BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets." Communications biology 3, no. 1 (2020): 103.

       Parameters
       ----------
       imgfix : str or array
           Path to the fixed map, or fixed array map.
       imgperm : str or array
           Path to the fixed map to be permuted, or array map to be permuted.
       nperm : int
           Number of permutations to perform.
       metric : str, optional
           Metric for comparing maps (one of pearsonr, spearmanr).
           Default is 'pearsonr'.
       label : str, optional
           Label for the hippocampus ('hipp' or 'dentate'). Default is 'hipp'.
       den : str, optional
           Density of the surface data. Default '0p5mm'.

       Returns
       -------
       metricnull : Null distribution of the specified metric
       permutedimg : All permuted spatial maps at 'unfoldiso' density.
       r_obs :  The observed association between the two aligned maps.
       pval : p-value based on metricnull r_obs.
       """
    if type(imgfix) == str:
        imgfix = nib.load(imgfix).agg_data().flatten()
    if type(imgperm) == str:
        imgperm = nib.load(imgperm).agg_data().flatten()
    
    # load reference surface to get geodesic distance
    surf = read_surface(f"{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{den}_label-{label}_midthickness.surf.gii")
    # wrap convenient brainspace function for weights (geodesic distance) and MRS
    weights = me.get_ring_distance(surf, n_ring=1)
    weights.data **= -1
    msr = MoranRandomization(n_rep=nperm, spectrum='all')
    msr.fit(weights)

    # get observed correlation
    r_obs = eval(metric)(imgfix, imgperm)[0]

    # randomize
    imgperm_rand = msr.randomize(imgperm)
    metricnull = np.ones((nperm))*np.nan
    for d in range(nperm):
        try:
            metricnull[d] = eval(metric)(imgfix, imgperm_rand[d,:])[0]
        except: 
            warnings.warn(f"permuation {d} contains a NaN or Inf")

    # p-value is the sum of all instances where null correspondance is >= observed correspondance / nperm
    pval = np.nanmean(np.abs(metricnull) >= np.abs(r_obs))  

    return metricnull, imgperm_rand, pval, r_obs

def eigenstrapping(imgfix, imgperm, nperm=1000, metric='pearsonr', label='hipp', den='0p5mm', num_modes=200, permute=False, resample=False, **qwargs):
    """
        Awesome new tool at https://www.biorxiv.org/content/10.1101/2024.02.07.579070v1.abstract
        Generates null models of spatial maps by rotating geometric eigenmodes.
        Koussis, N. C., Pang, J. C., Jeganathan, J., Paton, B., Fornito, A., Robinson, P. A., ... & Breakspear, M. (2024). Generation of surrogate brain maps preserving spatial autocorrelation through random rotation of geometric eigenmodes. bioRxiv, 2024-02.
       
        Parameters
        ----------
        imgfix : str or array
           Path to the fixed map, or fixed array map.
        imgperm : str or array
           Path to the fixed map to be permuted, or array map to be permuted.
        nperm : int
            Number of permutations to perform.
        metric : str, optional
            Metric for comparing maps (one of pearsonr, spearmanr).
            Default is 'pearsonr'.
        label : str, optional
            Label for the hippocampus ('hipp' or 'dentate'). Default is 'hipp'.
        den : str, optional
            Density of the surface data. Default '0p5mm'.
        num_modes : int, optional
            Number of eigenmodes to use. Default is 200.
        permute : bool, optional
            Set whether to permute surrogate map from original map to preserve values. Default is False.
        resample : bool, optional
            Set whether to resample surrogate map from original map to preserve values. Default is False.
        **qwargs : dict, optional
            Additional keyword arguments for customization.
            See https://eigenstrapping.readthedocs.io/en/latest/generated/eigenstrapping.SurfaceEigenstrapping.html#eigenstrapping.SurfaceEigenstrapping

        Returns
        -------
        metricnull : Null distribution of the specified metric
        permutedimg : All permuted spatial maps at 'unfoldiso' density.
        r_obs :  The observed association between the two aligned maps.
        pval : p-value based on metricnull r_obs.
        """
    if type(imgfix) == str:
        imgfix = nib.load(imgfix).agg_data().flatten()
    if type(imgperm) == str:
        imgperm = nib.load(imgperm).agg_data().flatten()
    
    # load reference surface and put in Eigenstrapping class
    eigen = SurfaceEigenstrapping(surface=f"{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-{den}_label-{label}_midthickness.surf.gii", 
        data=imgperm, num_modes=num_modes, permute=False, resample=False, **qwargs)

    # get observed correlation
    r_obs = eval(metric)(imgfix, imgperm)[0]

    # randomize
    hippomaps.utils.blockPrint()
    imgperm_rand = eigen(n=nperm)
    hippomaps.utils.enablePrint()
    metricnull = np.ones((nperm))*np.nan
    for d in range(nperm):
        try:
            metricnull[d] = eval(metric)(imgfix, imgperm_rand[d,:])[0]
        except: 
            warnings.warn(f"permuation {d} contains a NaN or Inf")

    # p-value is the sum of all instances where null correspondance is >= observed correspondance / nperm
    pval = np.nanmean(np.abs(metricnull) >= np.abs(r_obs))  

    return metricnull, imgperm_rand, pval, r_obs

def contextualize2D(taskMaps, n_topComparison=3, nperm=1000, plotTable=True, plot2D=True):
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
    if taskMaps.ndim==1: taskMaps = taskMaps.reshape([-1,1])
    nT = taskMaps.shape[1]

    # load required data
    contextHM = np.load(f'{resourcesdir}/2Dcontextualize/initialHippoMaps.npz')
    
    # resample all input data to 0p5mm (if needed)
    nV,iV = hippomaps.config.get_nVertices(['hipp'],'0p5mm')
    if taskMaps.shape[0] != nV:
        taskMapsresamp,_,_ = hippomaps.utils.density_interp(hm.config.get_label_from_nV(taskMaps.shape[0])[1],'0p5mm',taskMaps, label='hipp')
    else:
        taskMapsresamp = taskMaps

    # permute the input task maps (this is generally faster than permuting the reference features!)
    permutedTasks = np.zeros((nV,nT,nperm))
    metricnull = np.ones((nT,nperm))*np.nan
    hippomaps.utils.blockPrint()
    for t in range(nT):
            eigen = SurfaceEigenstrapping(surface=f"{resourcesdir}/canonical_surfs/tpl-avg_space-canonical_den-0p5mm_label-hipp_midthickness.surf.gii", 
                data=taskMapsresamp[:,t])
            permutedTasks[:,t,:] = eigen(nperm).T
    hippomaps.utils.enablePrint()
    # approx 9m to here

    # compute correlations with AP and all subfield permutations
    APR = np.ones((nT))*np.nan
    APp = np.ones((nT))*np.nan
    Subsp = np.ones((nT))*np.nan
    SubsR = np.ones((nT))*np.nan
    for t in range(nT):
        APR[t] = pearsonr(taskMapsresamp[:,t],contextHM['AP'])[0]
        SubsWithTask = pd.DataFrame(np.concatenate((taskMapsresamp[:,t].reshape([nV,1]),contextHM['subfields_permuted']),axis=1))
        Subfscorr = SubsWithTask.corr('spearman').to_numpy()[0,1:]
        Subfscorr_best = np.nanargmax(np.abs(Subfscorr))
        SubsR[t] = Subfscorr[Subfscorr_best]
        metricnull = np.zeros((nperm))
        for d in range(nperm):
            metricnull[d] = pearsonr(contextHM['AP'], permutedTasks[:,t,d])[0]
        APp[t] = np.nanmean(np.abs(metricnull) >= np.abs(APR[t]))
        metricnull = np.zeros((nperm))
        for d in range(nperm):
            metricnull[d] = spearmanr(contextHM['subfields_permuted'][:,Subfscorr_best], permutedTasks[:,t,d])[0]
        Subsp[t] = np.nanmean(np.abs(metricnull) >= np.abs(SubsR[t]))
    # 18m to here

    # compute correlations with extant features (but only the top 3 to save time)
    topFeatures = np.ones((nT,n_topComparison), dtype='object')
    topR = np.ones((nT,n_topComparison))*np.nan
    topP = np.ones((nT,n_topComparison))*np.nan
    if n_topComparison >0:
        p = np.zeros((nT,len(contextHM['features'])))
        R = np.zeros((nT,len(contextHM['features'])))
        for t in range(nT):
            for j in range(len(contextHM['features'])):
                R[t,j],p[t,j] = pearsonr(taskMapsresamp[:,t],contextHM['featureData'][:,j])
        # get ordering of the closest n_topComparison neighbours and then run spin test
        for t in range(nT):
            order = np.argsort(np.abs(R[t,:]))[::-1]
            for c in range(n_topComparison):
                topFeatures[t,c] = contextHM['features'][order[c]]
                topR[t,c] = R[t,order[c]]
                metricnull = np.zeros((nperm))
                for d in range(nperm):
                    metricnull[d] = pearsonr(contextHM['featureData'][:,order[c]], permutedTasks[:,t,d])[0]
                topP[t,c] = np.nanmean(np.abs(metricnull) >= np.abs(topR[t,c]))

    # generate dict of results
    context2D = {"Input map": list(range(nT)), 
        "Subfields (R)": SubsR, 
        "Subfields (p corrected)": Subsp, 
        "A-P gradinet (R)": APR,
        "A-P gradinet (p corrected)": APp}
    for c in range(n_topComparison):
        context2D[str(c) +"th most correlated"] = topFeatures[:,c]
        context2D[str(c) +"th (R)"] = topR[:,c]
        context2D[str(c) +"th (p corrected)"] = topP[:,c]
    
    if plotTable:
        print(tabulate(context2D,headers="keys"))

    fig, ax = plt.subplots(figsize=(8,8))
    if plot2D:
        ax.spines[['right', 'top']].set_visible(False)
        ax.scatter(contextHM['axiscorrAPPD'][0],contextHM['subfieldsmaxcorr'],c=contextHM['colors'],cmap='Set3',s=200)
        plt.ylabel("absolute subfield correlation (Spearmann's R)")
        plt.xlabel("absolute AP correlation (Pearson's R)")
        for f,feature in enumerate(contextHM['features']):
            ax.annotate(str(int(contextHM['feature_n'][f])), (contextHM['axiscorrAPPD'][0,f]-.008, contextHM['subfieldsmaxcorr'][f]-.007))
        ax.scatter(np.abs(APR),np.abs(SubsR),color='k',s=200);
        for t in range(nT):
            ax.annotate(str(t), (np.abs(APR[t])-.008, np.abs(SubsR[t])-.007),color='w')
            
    return context2D, ax
