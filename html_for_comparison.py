import random 
import os
import numpy as np
import pdb
import sys
import glob


# gt_dir = '/CT/nerf_hands/work/datasets/InterHand2.6M_submission_full/test_capture0/val/'
gt_dir = '/CT/nerf_hands/work/datasets/simulated_htmlTex/val/'

# exps = {
#         'wo mesh guided sampling': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp101_7_ablations_2levelSampling_test_capture0/quants140000',
#         'wo SR': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp101_4_ablations_woSR_test_capture0/quants340000',
#         'wo SR, LPIPS': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp101_5_ablations_woSR_woLPIPS_test_capture0/quants460000',
#         'ours': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_1_test_capture0/quants200000',
#         }

# exps = {
#         'iden1': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_1_test_capture0/quants200000',
#         'iden2': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_2_test_capture1/iden1_param_renderings',
#         'iden3': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_4_train_capture0/iden1_param_renderings',
#         'iden4': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_5_train_capture5/iden1_param_renderings',
#         }

# exps = {
#         'explicit': '/CT/nerf_hands/work/experiments/mesh_wrapping',
#         'anerf': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp100_1_anerf_test_capture0/quants100000',
#         'lisa': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp100_1_lisa_test_capture0/quants160000',
#         'smplpix': '/CT/nerf_hands/work/smplpix/interhands_training/smplpix_logs6_best/renders_val',
#         'ours': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_1_test_capture0/quants200000_final',
#         }

# exps = {
#         'ours': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp99_1_test_capture0/quants200000_final',
#         'smplpix': '/CT/nerf_hands/work/smplpix/interhands_training/smplpix_logs6_best/renders_val'
#         }

exps = {
        'anerf': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_2_anerf_simulated_HTMLtex/quants080000',
        'lisa': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_1_lisa_simulated_HTMLtex/quants140000',
        'smplpix': '/CT/nerf_hands/work/smplpix/syntheticHTML_logs1/renders_val',
        'ours': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp109_1_tc0_simulated_HTMLtex/quants220000',
        }

# exps = {
#         'no_noise': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp109_1_tc0_simulated_HTMLtex/quants220000',
#         '0.001': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_1_tc0_simulated_HTMLtex_noise0.001/quants240000',
#         '0.002': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_2_tc0_simulated_HTMLtex_noise0.002/quants220000', 
#         '0.003': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_3_tc0_simulated_HTMLtex_noise0.003/quants240000',
#         '0.005': '/CT/nerf_hands/work/experiments/nerf-pytorch/exp110_4_tc0_simulated_HTMLtex_noise0.005/quants220000'
#         }

filePath = '/CT/nerf_hands/work/experiments/comparisons/syntheicHTML_comparison.html'
f = open(filePath, 'w')
f.write('<html>\n<head><title>pose-noise</title></head>\n<body>\n')
f.write('<table style="text-align: center; margin-left: auto; margin-right: auto;" frame="border" border="5" cellpadding="1" cellspacing="1" width="0">\n')
f.write('<caption><h3>Results</h3></caption>\n<tr>')

f.write('\n<td> idx </td>')
for exp in exps.keys():
    f.write('\n<td>' + exp + '</td>')
f.write('\n<td>' + 'GT' + '</td>')
f.write('\n</tr>\n')


files = [f for f in glob.glob(exps['anerf'] + "/**/*.png", recursive=True)]
selected_files = random.sample(files, 100)

for i, file in enumerate(selected_files):

    filename = os.path.basename(file)
    dirname = os.path.basename(os.path.dirname(file))
    
    line = '<tr>\n<td>'
    line += f'{dirname}/{filename}' +'</td>\n<td>'
    
    for exp_name in exps.keys():
        exp_path = exps[exp_name]
        outPath = exp_path + '/' + dirname + '/' + filename
        line += '<img src= "'     + outPath     + '" height="512" border="0">' +'</td>\n<td>'
    
    gt_path = gt_dir + dirname + '/images/' + filename
    line += '<img src= "'     + gt_path     + '" height="512" border="0" style="background-color:black">' +'</td>\n<td>'
    
    line += '</tr>\n'
    f.write(line)