""" Example run od the GMM augmentation """

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import argparse
import os

from augmentation_utils import generate_gmm_image, normalize_image

parser = argparse.ArgumentParser(description='GMM-based augmentation')

parser.add_argument('-i',
                    '--img_dir',
                    help='Image_dir',
                    type=str,
                    default='data/t1.nii.gz')

parser.add_argument('-m',
                    '--mask_dir',
                    help='Mask dir',
                    type=str,
                    default='data/mask.nii.gz')

parser.add_argument('-n',
                    '--n_components',
                    type=int,
                    default=3
                    )

parser.add_argument('-mu',
                    '--std_means',
                    nargs='+',
                    type=float,
                    default=(0.03, 0.06, 0.08)
                    )

parser.add_argument('-s',
                    '--std_sigma',
                    nargs='+',
                    type=float,
                    default=(0.012, 0.011, 0.015)
                    )

parser.add_argument('-p',
                    '--percentiles',
                    nargs='+',
                    type=float,
                    default=(1, 99)
                    )


def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise
        
        
def demo_gmm_augmentation(args):

    # Load the image and mask
    print('Loading image and mask...')
    t1_path = args.img_dir
    mask_path = args.mask_dir
    t1 = nib.load(t1_path)
    t1_data = t1.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    n_components = args.n_components
   
    # Clip percentiles and normalize the image
    t1_normalized = normalize_image(t1_data, clip_percentiles=False, pmin=args.percentiles[0], pmax=args.percentiles[1])

    print('Augmenting image...')
    new_image = generate_gmm_image(t1_normalized, mask=mask,
                                   n_components=n_components,
                                   std_means=tuple(args.std_means),
                                   std_sigma=tuple(args.std_sigma),
                                   p_mu=None, q_sigma=None,
                                   normalize=False)
    
    # Save the augmented image
    print('Saving augmented image...')
    create_dir('results/')
    nib.save(nib.Nifti1Image(new_image, t1.affine, header=t1.header), 'results/augmented_img.nii.gz')
    
    # Plot and save the figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    ax[0].imshow(t1_normalized[:, :, int(np.shape(t1_normalized)[2] / 2)].T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Original image')
    
    ax[1].imshow(new_image[:, :, int(np.shape(new_image)[2] / 2)].T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('Augmented image')
    plt.savefig('results/demo.png')


if __name__ == "__main__":
    demo_gmm_augmentation(parser.parse_args())
