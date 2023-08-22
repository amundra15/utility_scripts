import numpy as np
import cv2
import pdb
import os




# pose noise
frames = [
        'frame23308/15.png' 	
        ]


#define standard colours
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


# specify the folders and optionally the highlight colours
# exp_folders = {
#         '/CT/nerf_hands/work/smplpix/syntheticHTML_logs1/renders_val': green,
#         '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_2_anerf_simulated_HTMLtex/quants080000': red,
#         '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_1_lisa_simulated_HTMLtex/quants140000': red,
#         '/CT/nerf_hands/work/experiments/nerf-pytorch/exp109_1_tc0_simulated_HTMLtex/quants220000': green,
# }

exp_folders = {
        '/CT/nerf_hands/work/experiments/nerf-pytorch/exp109_1_tc0_simulated_HTMLtex/quants220000': None,
        '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_1_tc0_simulated_HTMLtex_noise0.001/quants240000': None,
        # '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_2_tc0_simulated_HTMLtex_noise0.002/quants220000': None,
        '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_3_tc0_simulated_HTMLtex_noise0.003/quants240000': None,
        '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_4_tc0_simulated_HTMLtex_noise0.005/quants220000': None,
}



out_path = '/CT/nerf_hands/work/experiments/comparisons/pose_noise'
os.makedirs(out_path, exist_ok=True)

# gt_dir = '/CT/nerf_hands/work/datasets/InterHand2.6M_submission_full/test_capture0/val/'
gt_dir = '/CT/nerf_hands/work/datasets/simulated_htmlTex/val/'






# Define your parameters for highlighting and zooming
highlight_box = {'x': 90, 'y': 280, 'width': 50, 'height': 50}
zoom_box = {'x': 50, 'y': 110, 'ratio': 2}

zoom_box['width'] = highlight_box['width']*zoom_box['ratio']
zoom_box['height'] = highlight_box['height']*zoom_box['ratio']


def highlight_and_zoom(image, highlight_box, zoom_box, highlight_colour):
    
    #highlight the box
    highlighted_img = image.copy()
    cv2.rectangle(highlighted_img, (highlight_box['x'], highlight_box['y']),
                    (highlight_box['x'] + highlight_box['width'], highlight_box['y'] + highlight_box['height']),
                    highlight_colour, 1)

    #zoom in on the side
    highlight_crop = image[highlight_box['y']:highlight_box['y'] + highlight_box['height'], highlight_box['x']:highlight_box['x'] + highlight_box['width']]
    zoomed_crop = cv2.resize(highlight_crop, (zoom_box['width'], zoom_box['height']))
    highlighted_img[zoom_box['y']:zoom_box['y'] + zoom_box['height'], zoom_box['x']:zoom_box['x'] + zoom_box['width']] = zoomed_crop
    
    #highlight the zoomed box
    cv2.rectangle(highlighted_img, (zoom_box['x'], zoom_box['y']),
                (zoom_box['x'] + zoom_box['width'], zoom_box['y'] + zoom_box['height']),
                highlight_colour, 1)
    
    return highlighted_img



# #concat the images horizontally and then put them together vertically
# hor_concats = []

# for frame in frames:
#     print("frame: ", frame)
#     images = []
    
    
#     #load and concatenate the images
#     for exp_folder in exp_folders:
#         img = cv2.imread(exp_folder + '/' + frame)
#         if exp_folders[exp_folder] is not None:
#             img = highlight_and_zoom(img, highlight_box, zoom_box, exp_folders[exp_folder])
#         images.append(img)
    
    
#     gt_path = gt_dir + frame.split("/")[0] + '/images/' + frame.split("/")[1]
#     gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
#     mask = gt[:,:,3] == 0
#     gt = gt[:,:,:3] * (1 - mask[:,:,np.newaxis]) + mask[:,:,np.newaxis] * 0
#     gt = gt.astype(np.uint8)
#     gt = highlight_and_zoom(gt, highlight_box, zoom_box, blue)
#     images.append(gt)

#     img = np.concatenate(images, axis=1)
    
    
#     save_name = f'{frame.split("/")[0]}_{frame.split("/")[1]}'
#     cv2.imwrite(out_path + '/' + save_name, img)
    
#     # hor_concats.append(img)

# # img = np.concatenate(hor_concats, axis=0)
# # cv2.imwrite(out_path + '/all.png', img)



#concat the images vertically and then put them together horizontally
ver_concats = []

for frame in frames:

    print("frame: ", frame)

    images = []
    crop_top = 110
    crop_bottom = 120
        
    gt_path = gt_dir + frame.split("/")[0] + '/images/' + frame.split("/")[1]
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    mask = gt[:, :, 3] == 0
    gt = gt[:, :, :3] * (1 - mask[:, :, np.newaxis]) + mask[:, :, np.newaxis] * 0
    gt = gt.astype(np.uint8)
    # gt = highlight_and_zoom(gt, highlight_box, zoom_box, blue)
    images.append(gt[crop_top:-crop_bottom, :, :])


    for exp_folder in exp_folders:
        img = cv2.imread(exp_folder + '/' + frame)
        if exp_folders[exp_folder] is not None:
            img = highlight_and_zoom(img, highlight_box, zoom_box, exp_folders[exp_folder])
        images.append(img[crop_top:-crop_bottom, :, :])

    img = np.concatenate(images, axis=0)

    # crop_top = 100
    # crop_bottom = 100
    # img = img[crop_top:-crop_bottom, :, :]

    save_name = f'{frame.split("/")[0]}_{frame.split("/")[1]}'
    cv2.imwrite(out_path + '/' + save_name, img)

    ver_concats.append(img)

# img = np.concatenate(ver_concats, axis=1)
# cv2.imwrite(out_path + '/all.png', img)
