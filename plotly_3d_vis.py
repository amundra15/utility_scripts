import os
import sys
import numpy as np
import pdb
from termcolor import colored

import torch
from pytorch3d.io import load_obj
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mano_fitter_pytorch.smplx_extended as smplx_extended



#get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'InterHand2.6M'
# dataset = 'Hand3Dstudio'
#create MANO layer
smplx_path = '../mano_fitter_pytorch/SMPLX_models/models'
if dataset == 'InterHand2.6M':
    use_flat_hand_mean = False
    mano_scale = 1
elif dataset == 'Hand3Dstudio':
    use_flat_hand_mean = True
    mano_scale = 15
    print(colored('WARNING: Verify if we should use flat hand mean for Hand3Dstudio', 'red'))
mano_layer = {
            'right': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=True, num_pca_comps=45, is_Euler=False, flat_hand_mean=use_flat_hand_mean, scale=mano_scale),
            'left': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=False, num_pca_comps=45, is_Euler=False, flat_hand_mean=use_flat_hand_mean, scale=mano_scale)
            }
if 'cuda' in device.type:
    mano_layer['right'].cuda()
    mano_layer['left'].cuda()

faces = mano_layer['right'].faces


# dataset_folder = '/CT/nerf_hands/work/datasets/InterHand2.6M_5fps/parsed_1iden_allPose/capture0_subject10_test_allAnnot'
dataset_folder = '/CT/nerf_hands/work/datasets/InterHand2.6M_submission_full/test_capture0/train/'


#code to visualise intermediate results for a single frame
if False:
    
    data_fol = '/CT/nerf_hands/work/datasets/Hand3Dstudio/parsed_1iden_allPose/val/00_index111/mesh/'
    intermediates_path = os.path.join(data_fol, 'intermediates')
    # reconstructed_mesh = load_obj(os.path.join(data_fol, 'point_cloud.obj'), load_textures=False, device=device)
    reconstructed_mesh = load_obj(os.path.join(data_fol, 'mesh.obj'), load_textures=False, device=device)

    frames = []
    num_iters = 10000
    # for i in range(0, num_iters, num_iters//10):
    for i in range(0, num_iters, 500):
        
        filename = os.path.join(intermediates_path, f'iter_{i}.pt')
        saved_params = torch.load(filename, map_location=device)
        
        #forward pass
        mano_output = mano_layer[saved_params['hand_type']](global_orient=saved_params['root_pose'], 
                                                        hand_pose=saved_params['hand_pose'], 
                                                        betas=saved_params['shape_param'], 
                                                        transl=saved_params['root_trans'])
        vertices = mano_output.vertices.detach().cpu().numpy()
        data = go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink', opacity=0.5)
        
        loss_value = saved_params['loss']
        
        #animated 3d mesh plotly figure
        button = {
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 10}}],
                }
            ],
        }

        layout = go.Layout(updatemenus=[button], title_text=f"Iter: {i}/{num_iters}, Loss: {loss_value: .8f}")
        frame = go.Frame(data=[data], layout=layout)
        frames.append(frame)

    # combine the graph_objects into a figure
    fig = go.Figure(data=[data], frames=frames, layout=layout)

    # add the dense point cloud
    fig.add_trace(go.Scatter3d(x=reconstructed_mesh[0][:, 0].detach().cpu().numpy(), 
                            y=reconstructed_mesh[0][:, 1].detach().cpu().numpy(), 
                            z=reconstructed_mesh[0][:, 2].detach().cpu().numpy(), 
                            mode='markers', marker=dict(size=1, color='blue')))
        
    fig.show()


#code to visualise final results for all frames
if True:
    
    subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir() and f.name.startswith('frame') and os.path.exists(os.path.join(f.path, 'neus_fitting', 'MANO_params.pkl'))]
    # subfolders = [os.path.join(dataset_folder, 'frame19028')]
    print(f'Found {len(subfolders)} frames')
    
    for i, subfolder in enumerate(subfolders):
        
        if i > 20:
            continue
        
        fig = go.Figure()
        
        #load the files
        ori_fitting = np.load(os.path.join(subfolder, 'mesh', 'MANO_params.pkl'), allow_pickle=True)
        
        # metashape_pointcloud = load_obj(os.path.join(subfolder, 'metashape_fitting', 'point_cloud.obj'), load_textures=False, device=device)
        # metashape_fitting = torch.load(os.path.join(subfolder, 'metashape_fitting', 'MANO_params.pt'), map_location=device)
        
        neus_pointcloud = load_obj(os.path.join(subfolder, 'neus_fitting', 'mesh.obj'), load_textures=False, device=device)
        neus_fitting = np.load(os.path.join(subfolder, 'neus_fitting', 'MANO_params.pkl'), allow_pickle=True)
        

        #show the dense point cloud
        fig.add_trace(go.Scatter3d(x=neus_pointcloud[0][:, 0].detach().cpu().numpy(),
                                y=neus_pointcloud[0][:, 1].detach().cpu().numpy(), 
                                z=neus_pointcloud[0][:, 2].detach().cpu().numpy(), 
                                mode='markers', marker=dict(size=1, color='blue')))
        
        
        #show original MANO fitting
        root_pose = torch.from_numpy(ori_fitting['pose'][:, :3]).float().to(device)
        hand_pose = torch.from_numpy(ori_fitting['pose'][:, 3:]).float().to(device)
        shape = torch.from_numpy(ori_fitting['shape']).float().to(device)
        trans = torch.from_numpy(ori_fitting['trans']).float().to(device)
        mano_output = mano_layer[ori_fitting['hand_type']](global_orient=root_pose,
                                                        hand_pose=hand_pose,
                                                        betas=shape,
                                                        transl=trans)
        vertices = mano_output.vertices.detach().cpu().numpy()
        fig.add_trace(go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink', opacity=0.5))
        
        
        
        # #show NeUS2 based fitting
        # root_pose = torch.from_numpy(neus_fitting['root_pose']).float().to(device)
        # hand_pose = torch.from_numpy(neus_fitting['hand_pose']).float().to(device)
        # shape = torch.from_numpy(neus_fitting['shape_param']).float().to(device)
        # trans = torch.from_numpy(neus_fitting['root_trans']).float().to(device)
        # mano_output = mano_layer[neus_fitting['hand_type']](global_orient=root_pose,
        #                                                     hand_pose=hand_pose,
        #                                                     betas=shape,
        #                                                     transl=trans)
        # vertices = mano_output.vertices.detach().cpu().numpy()
        # data = go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightgreen', opacity=0.5)     #coral
        # fig.add_trace(data)
        
        
        
        # #show metashape fitting
        # mano_output = mano_layer[metashape_fitting['hand_type']](global_orient=metashape_fitting['root_pose'],
        #                                             hand_pose=metashape_fitting['hand_pose'],
        #                                             betas=metashape_fitting['shape_param'],
        #                                             transl=metashape_fitting['root_trans'])
        # vertices = mano_output.vertices.detach().cpu().numpy()
        # fig.add_trace(go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink', opacity=0.5))
        
        # #show metashape point cloud
        # fig.add_trace(go.Scatter3d(x=metashape_pointcloud[0][:, 0].detach().cpu().numpy(),
        #                         y=metashape_pointcloud[0][:, 1].detach().cpu().numpy(),
        #                         z=metashape_pointcloud[0][:, 2].detach().cpu().numpy(),
        #                         mode='markers', marker=dict(size=1, color='orange')))
        

        #add frame number to the title
        fig.update_layout(title_text=f'Frame {subfolder[-5:]}')
        
        
        
        
        # #TEMP
        # min_hand_shape = torch.tensor([[-2.58199644, -0.07120129, -0.81127405,  0.01359154,  0.1027914, -0.00357446, 0.2475048,   0.06110404, -0.15868324, -0.04786261]]).float().to(device)
        # max_hand_shape = torch.tensor([[-0.23785305,  0.3192907,  -0.19199118,  0.02058078, -0.0187702,  -0.1100082,  -0.17300454,  0.21638456, -0.00532272, -0.00782738]]).float().to(device)
        
        # #show original MANO fitting with min and max hand shape values (0th entry of the PCA vector)
        # root_pose = torch.from_numpy(ori_fitting['pose'][:, :3]).float().to(device)
        # hand_pose = torch.from_numpy(ori_fitting['pose'][:, 3:]).float().to(device)
        # shape = torch.from_numpy(ori_fitting['shape']).float().to(device)
        # trans = torch.from_numpy(ori_fitting['trans']).float().to(device)
        # mano_output = mano_layer[ori_fitting['hand_type']](global_orient=root_pose,
        #                                                 hand_pose=hand_pose,
        #                                                 betas=min_hand_shape,
        #                                                 transl=trans)
        # vertices = mano_output.vertices.detach().cpu().numpy()
        # fig.add_trace(go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink', opacity=0.5))
        
        # mano_output = mano_layer[ori_fitting['hand_type']](global_orient=root_pose,
        #                                                 hand_pose=hand_pose,
        #                                                 betas=max_hand_shape,
        #                                                 transl=trans)
        # vertices = mano_output.vertices.detach().cpu().numpy()
        # fig.add_trace(go.Mesh3d(x=vertices[0,:,0], y=vertices[0,:,1], z=vertices[0,:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightblue', opacity=0.5))
        
        
        fig.show()