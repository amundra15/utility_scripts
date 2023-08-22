import numpy as np
import pdb
import os
import cv2
from termcolor import colored


# img2mse = lambda x, y : torch.mean((x - y) ** 2)
# mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def psnr(im1, im2):
    mse = np.mean((im1.astype(np.float64) / 255 - im2.astype(np.float64) / 255) ** 2)
    psnr_val = -10 * np.log10(mse)
    return psnr_val


def error_vis(im1, gt, display_PSNR=False):
    
    if gt.shape[2] == 4:
        mask = gt[:,:,3] > 127
        mask = np.dstack((mask, mask, mask))
        gt = gt[:, :, :3] * mask

    error = np.abs((im1 - gt))
    error = np.mean(error, axis=2)
    # error = (error - np.min(error)) / (np.max(error) - np.min(error))
    # error[error!=0] = (error - np.min(error[error != 0])) / (np.max(error) - np.min(error[error != 0]))
    error = np.uint8(error)
    error = cv2.applyColorMap(error, cv2.COLORMAP_JET)
    
    if display_PSNR:
        # psnr_val = psnr(im1*mask, gt*mask)
        psnr_val = cv2.PSNR(im1*mask, gt*mask)
        error = cv2.putText(error, f'PSNR: {psnr_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return error



# if __name__=='__main__':

#     im_path = '/CT/nerf_hands/work/experiments/exp80_3_real_config1_1i1p_texCondioning_noDepthBasedSampling/trainRes/013999_frame12966_cam42.png'
#     im1 = cv2.imread(im_path)

#     dataset_path = '/CT/nerf_hands/work/datasets/InterHand2.6M_5fps/parsed_1iden_allPose/capture0_subject10_test_allAnnot'
#     frame_name = im_path.split('/')[-1].split('_')[1]
#     cam_index = im_path.split('/')[-1].split('_')[2][3:-4]
#     gt_path = os.path.join(dataset_path, f'{frame_name}/images/{cam_index}.png')
#     gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

#     error_im = error_vis(im1, gt)

#     save_path = im_path[:-4] + '_error.png'
#     cv2.imwrite(save_path, error_im)