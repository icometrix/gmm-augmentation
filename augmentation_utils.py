"""
File containing auxiliary functions for GMM-based data augmentation.
"""

import numpy as np
from sklearn import mixture

def normalize_image(image, clip_percentiles=False, pmin=1, pmax=99):
    """
    Function to normalize the images between [0,1]. If percentiles is set to True it clips the intensities at
     percentile 1 and 99
    :param image: numpy array containing the image
    :param clip_percentiles: set to True to clip intensities. (default: False)
    :param pmin: lower percentile to clip
    :param pmax: upper percentile to clip
    :return: normalized image [0,1]
    """
    if clip_percentiles is True:
        pmin = np.percentile(image, pmin)
        pmax = np.percentile(image, pmax)
        v = np.clip(image, pmin, pmax)
    else:
        v = image.copy()

    v_min = v.min(axis=(0, 1, 2), keepdims=True)
    v_max = v.max(axis=(0, 1, 2), keepdims=True)

    return (v - v_min) / (v_max - v_min)


def select_component_size(x, min_components=1, max_components=5):
    """
    Function that selects the optimal number of components for a given image, based on the BIC criterion.
    :param x: non-zero values of image. Should be of shape (N,1). Make sure to use X=np.expand_dims(data(data>0),1)
    :param min_components: minimum number of components allowed for the gmm
    :param max_components: maximum number of components allowed for the gmm
    :return: optimal number of components for X
    """
    lowest_bic = np.infty
    bic = []
    n_components_range = range(min_components, max_components)
    cv_type = 'full'  # covariance type for the GaussianMixture function
    best_gmm = None
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(x)
        bic.append(gmm.aic(x))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    return best_gmm.n_components, best_gmm


def fit_gmm(x, n_components=3):
    """ Fit the GMM to the data
    :param x:  non-zero values of image. Should be of shape (N,1). Make sure to use X=np.expand_dims(data(data>0),1)
    :param n_components: number of components in the mixture. Set to None to select the optimal component number based
           on the BIC criterion. Default: 3
    :return: GMM model, fit to the data
    """
    if n_components is None:
        n_components, gmm = select_component_size(x, min_components=3, max_components=7)
    else:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag', tol=1e-3)

    return gmm.fit(x)


def get_new_components(gmm, p_mu=None, q_sigma=None,
                       std_means=None, std_sigma=None):
    """ Computes the new intensity components by shifting the mean and variance of the individual components
    predicted by the GMM
    :param gmm: GMM model fit to
    :param p_mu: tuple containing the value by which to change the mean in each component in the mixture. Use only to
                 check the behaviour of the method or to get specific changes.
    :param q_sigma: tuple containing the value by which to change the standard deviation in each component in the mixture.
                    Use only to check the behaviour of the method or to get specific changes.
    :param std_means: tuple containing the range of variability to change the mean of each component in the
                      mixture. If set to None, all components are augmented in a similar way
    :param std_sigma: tuple containing the range of variability to change the standard deviation of each component in
                      the mixture. If set to None, all components are augmented in a similar way
    :return dictionary containing original and updated means and standard deviations for each component
    """

    sort_indices = gmm.means_[:, 0].argsort(axis=0)
    mu = np.array(gmm.means_[:, 0][sort_indices])
    std = np.array(np.sqrt(gmm.covariances_[:, 0])[sort_indices])
    
    n_components = mu.shape[0]
   
    if std_means is not None:
        # use pre-computed intervals to draw values for each component in the mixture
        rng = np.random.default_rng()
        if p_mu is None:
            var_mean_diffs = np.array(std_means)
            p_mu = rng.uniform(-var_mean_diffs, var_mean_diffs)
    if std_sigma is not None:
        rng = np.random.default_rng()
        if q_sigma is None:
            var_std_diffs = np.array(std_sigma)
            q_sigma = rng.uniform(-var_std_diffs, var_std_diffs)
    else:
        # Draw random values for each component in the mixture
        # Multiply by random int for shifting left (-1), right (1) or not changing (0) the parameter.
        if p_mu is None:
            p_mu = 0.06 * np.random.random(n_components) * np.random.randint(-1, 2, n_components)
        if q_sigma is None:
            q_sigma = 0.005 * np.random.random(n_components) * np.random.randint(-1, 2, n_components)

    new_mu = mu + p_mu
    new_std = std + q_sigma
    
    return {'mu': mu, 'std': std, 'new_mu': new_mu, 'new_std': new_std}


def reconstruct_intensities(data, dict_parameters):
    """ Reconstruct the new image intensities from the new components.
    :param data: original image intensities (nonzero)
    :param dict_parameters: dictionary containing original and updated means and standard deviations for each component
    :return intensities_im: updated intensities for the new components
    """
    mu, std = dict_parameters['mu'], dict_parameters['std']
    new_mu, new_std = dict_parameters['new_mu'], dict_parameters['new_std']
    n_components = len(mu)
    
    # if we know the values of mean (mu) and standard deviation (sigma) we can find the new value of a voxel v
    # Fist we find the value of a factor w that informs about the percentile a given pixel belongs to: mu*v = d*sigma
    d_im = np.zeros(((n_components,) + data.shape))
    for k in range(n_components):
        d_im[k] = (data.ravel() - mu[k]) / (std[k] + 1e-7)

    # we force the new pixel intensity to lie within the same percentile in the new distribution as in the original
    # distribution: px = mu + d*sigma
    intensities_im = np.zeros(((n_components,) + data.shape))
    for k in range(n_components):
        intensities_im[k] = new_mu[k] + d_im[k] * new_std[k]

    return intensities_im


def get_new_image_composed(intensities_im, probas_original):
    """
    Compose the new image (brain)
    :param intensities_im: image intensities for the new components
    :param probas_original: initial probabilities for each component (as predicted by the GMM model)
    :return new_image_composed: new image after augmentation (skull stripped)
    """
    n_components = probas_original.shape[1]
    new_image_composed = np.zeros(intensities_im[0].shape)
    for k in range(n_components):
        new_image_composed = new_image_composed + probas_original[:, k] * intensities_im[k]

    return new_image_composed 


def generate_gmm_image(image, mask=None, n_components=None,
                       q_sigma=None,  p_mu=None,
                       std_means=None, std_sigma=None,
                       normalize=True, percentiles=True):
    """
    Funtion that takes an image and generates a new one by shifting the
    :param image: image to transform
    :param mask: brain mask.
    :param n_components: number of components in the mixture. if set to None will select best model based on AIC
    :param p_mu: tuple with same dimension as n_components containing factors to add to the mean of individual
                 components. (default: None)
    :param q_sigma: list with same dimension as n_components containing factors to add to the std of individual
                    components. (default: None)
    :param std_means: tuple containing the range of variability to change the mean of each component in the
                      mixture. If set to None, all components are augmented in a similar way
    :param std_sigma: tuple containing the range of variability to change the standard deviation of each component in
                      the mixture. If set to None, all components are augmented in a similar way
    :param normalize: if set to True will clip intensities at percentile 1 and 99 normalize the images between [0,1]
    :param percentiles: set to True to clip at percentiles 1 and 99.
    :return: new_image: image with updated intensities inside the brain mask, normalized to [0,1]
    """
        
    if normalize:
        image = normalize_image(image, clip_percentiles=percentiles, pmin=1, pmax=99)  # the percentiles can be changed

    if mask is None:
        masked_image = image
    else:
        masked_image = image * mask

    # # we only want nonzero values
    data = masked_image[masked_image > 0]
    x = np.expand_dims(data, 1)

    gmm = fit_gmm(x, n_components)
    sort_indices = gmm.means_[:, 0].argsort(axis=0)

    # Estimate the posterior probabilities
    probas_original = gmm.predict_proba(x)[:, sort_indices]

    # Get the new intensity components
    params_dict = get_new_components(gmm, p_mu=p_mu, q_sigma=q_sigma,
                                     std_means=std_means, std_sigma=std_sigma)
    intensities_im = reconstruct_intensities(data, params_dict)

    # Then we add the three predicted images by taking into consideration the probability that each pixel belongs to a
    # certain component of the gaussian mixture (probas_original)
    new_image_composed = get_new_image_composed(intensities_im, probas_original)

    # Reconstruct the image
    new_image = np.zeros(image.shape)
    new_image[np.where(masked_image > 0)] = new_image_composed

    # Put the skull back
    new_image[np.where(masked_image == 0)] = image[np.where(masked_image == 0)]

    # Return the image in [0,1]
    new_image = normalize_image(new_image, clip_percentiles=False)
    
    return new_image
