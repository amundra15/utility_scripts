import random
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


exps_dir = os.path.expanduser('~/networks/projects_data/AE2302_STADTUP/Datasets/instructpix2pix/cityscapes/v1.2/')
exps = {
        'snowy': f'{exps_dir}/snowy',
        'rainy': f'{exps_dir}/rainy',
        }

gt_dir = os.path.join(exps_dir, 'gt')
vis_gt = True


files = [f for f in glob.glob(exps['rainy'] + "/**/*.png", recursive=True)]
selected_files = random.sample(files, 100) if len(files) > 100 else files

for exp_name, exp_folder in exps.items():
    
    print(f'Generating plot for {exp_name}...')

    filePath = f'{exps_dir}/plot_{exp_name}.png'

    cfgtexts = []
    cfgimages = []
    rgbs = []
    gts = []
    

    for file in selected_files:
        
        file = file.replace(exps['rainy'], exps[exp_name])
        img = Image.open(file)
        rgbs.append(img)
        
        # Parse cfgtext and cfgimage from the filename
        info = img.info
        cfgtexts.append(float(info['cfg_text']))
        cfgimages.append(float(info['cfg_image']))
        
        if vis_gt:
            relative_parents = os.path.relpath(os.path.dirname(file), exp_folder)
            gt_path = os.path.join(gt_dir, relative_parents, os.path.basename(file))
            gts.append(Image.open(gt_path))

    # plot the exp fig
    exp_title = os.path.relpath(exp_folder, exps_dir)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(cfgtexts, cfgimages)
    for cfgtext, cfgimage, rgb in zip(cfgtexts, cfgimages, rgbs):
        ab = AnnotationBbox(OffsetImage(rgb, zoom=0.35), (cfgtext, cfgimage), frameon=False)
        ax.add_artist(ab)
    plt.xlabel('cfgtext', fontsize=30)
    plt.ylabel('cfgimage', fontsize=30)
    plt.title(exp_title, fontsize=30, y=1.05)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(filePath, dpi=300)


    # plot the gt fig
    if vis_gt:
        print(f'Generating GT reference plot for {exp_name}...')
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.scatter(cfgtexts, cfgimages)
        for cfgtext, cfgimage, gt in zip(cfgtexts, cfgimages, gts):
            ab = AnnotationBbox(OffsetImage(gt, zoom=0.35), (cfgtext, cfgimage), frameon=False)
            ax.add_artist(ab)
        plt.xlabel('cfgtext', fontsize=30)
        plt.ylabel('cfgimage', fontsize=30)
        plt.title(f'GT (only for reference for {exp_name})', fontsize=30, y=1.05)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'{exps_dir}/plot_{exp_name}_gt.png', dpi=300)