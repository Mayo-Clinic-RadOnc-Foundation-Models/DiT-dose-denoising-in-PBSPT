# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion_con 
from diffusers.models import AutoencoderKL
from download import find_model

from models.models import DiT_models
import argparse
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from scipy.io import savemat



def find_patplan_with_pat(strings, substring):
    for idx, string in enumerate(strings):
        if substring in string:
            return idx, string

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    modelxy = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    

    ckpt_pathxy = './results/checkpoints/weights.pt'

    xymodelname = 'folder'+(ckpt_pathxy.split('/')[2]).split('-')[0]+'epoch'+(ckpt_pathxy.split('/')[-1]).split('.')[0]
    state_dict = find_model(ckpt_pathxy)
    modelxy.load_state_dict(state_dict, strict = False)



    test_pat= ['xxx']
    ture_mean = np.zeros((10,))
    pred_mean = np.zeros((10,))
    ture_std = np.zeros((10,))
    pred_std = np.zeros((10,))
    cc = 0
    for pat in test_pat:
        saved_folder = './'+pat
        folder_path = './result_sample_img/'+saved_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        model.eval()  # important!
        modelxy.eval()
        diffusion = create_diffusion_con(str(args.num_sampling_steps))


        test_samples = np.load('./data_list_pkl/dose/sample_list_test4x32x32.pkl', allow_pickle=True)
               
        first_idx,fullname = find_patplan_with_pat(test_samples, pat)



        name = fullname.split('_chunk')[0]
        print(name)
        total_num_samples = 0
        for i in range(len(test_samples)):
            if i+first_idx ==len(test_samples):
                break
            elif name in test_samples[i+first_idx]:
                total_num_samples = total_num_samples+1     
            else:
                break
        num_samples = min(200, total_num_samples)
         

        original_shape = (fullname.split('size')[1]).split('_crop')[0]
        zshape, yshape, xshape = int(original_shape.split('_')[0]), int(original_shape.split('_')[1]), int(original_shape.split('_')[2])
        cropsize = (fullname.split('cropsize')[1]).split('_chunk')[0]
        cropzmin,cropzmax,cropymin,cropymax,cropxmin,cropxmax = int(cropsize.split('_')[0]),int(cropsize.split('_')[1]),int(cropsize.split('_')[2]),int(cropsize.split('_')[3]),int(cropsize.split('_')[4]),int(cropsize.split('_')[5])


        samplesallxycrop = torch.zeros(((cropzmax-cropzmin)*(cropymax-cropymin)*(cropxmax-cropxmin)))
        y_trueallcrop = torch.zeros(((cropzmax-cropzmin)*(cropymax-cropymin)*(cropxmax-cropxmin)))
        x_conallcrop = torch.zeros(((cropzmax-cropzmin)*(cropymax-cropymin)*(cropxmax-cropxmin)))

        for idx in range(int(np.ceil(total_num_samples/num_samples))):
             
            start_idx = idx * num_samples
            end_idx = start_idx + num_samples

            # Adjust for the last batch
            if end_idx > total_num_samples:
                end_idx = total_num_samples
                

        
            z = torch.randn(end_idx - start_idx, 4, latent_size, latent_size, device=device)
            ct = []
            x_con = []
            y_true = []

            
            for k in range(end_idx - start_idx):
                img = np.load('../dose4x32x32/'+test_samples[first_idx+k+start_idx]+'all.npy')
                ct.append(img[0,:])
                tmp = img[1,:]
                x_con.append(np.log1p(tmp))
                # x_con.append(np.log(1+ tmp*25))
                tmp = img[2,:]
                y_true.append(np.log1p(tmp))
                # y_true.append(np.log(1+ tmp*25))

             
            ct = torch.from_numpy(np.array(ct)).type(torch.FloatTensor)
            x_con = torch.from_numpy(np.array(x_con)).type(torch.FloatTensor)
            y_true = torch.from_numpy(np.array(y_true)).type(torch.FloatTensor)


            ct[ct<0] = 0   #### correct data

            ct = ct.to(device)
            x_con = x_con.to(device)
            y_true = y_true.to(device)


            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)

            model_kwargs = dict(cfg_scale=args.cfg_scale, ct=ct)

            samplesxy = diffusion.p_sample_loop(modelxy.forward_with_cfg, z.shape, z, x_con, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
            

            samplesxy[samplesxy<0] = 0

            samplesxy, _ = samplesxy.chunk(2, dim=0)
            print('dose' )   
            print('y_true min, max, mean and std: '+str(y_true.min())+' '+str(y_true.max())+' '+str(y_true.mean())+' '+str(y_true.std()))            
            print('y_pre (xy) min, max, mean and std: '+str(samplesxy.min())+' '+str(samplesxy.max())+' '+str(samplesxy.mean())+' '+str(samplesxy.std()))



  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XS/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
