from __future__ import print_function
import numpy as np
import torch
import cv2
import yaml
import os
from torch.autograd import Variable
from models.networks import get_generator
import torchvision
import time
import argparse
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser('Test an image')
    parser.add_argument('--job_name', default='DeblurDiNATL',
                        type=str, help='current job s name')
    parser.add_argument('--blur_path', default='/content/flat_directory',
                        type=str, help='blurred image path')
    parser.add_argument('--weight_name', default='DeblurDiNATL.pth',
                        type=str, help='pre-trained weights path')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Load configuration
    with open(os.path.join('config', args.job_name, 'config_custom_val.yaml')) as cfg:
        config = yaml.safe_load(cfg)
    
    blur_path = args.blur_path
    out_path = os.path.join('results', args.job_name, 'images')
    weights_path = os.path.join('results', args.job_name, 'models', args.weight_name)
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    # Load model
    model = get_generator(config['model'])
    ck = torch.load(weights_path, weights_only=True)
    new_state_dict = {key.replace("module.", ""): val for key, val in ck.items()}
    model.load_state_dict(new_state_dict)
    
    model = model.cuda()


    test_time = 0
    iteration = 0

    # Hardware warm-up
    warm_up = 0
    print('Hardware warm-up')
    for img_name in os.listdir(blur_path):
        if img_name.endswith(('png', 'jpg', 'jpeg')):  # Process only image files
            warm_up += 1
            img = cv2.imread(os.path.join(blur_path, img_name))
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
                result_image = model(img_tensor)
            if warm_up == 20:
                break

    # Process each image in the flat directory
    for img_name in os.listdir(blur_path):
        if img_name.endswith(('png', 'jpg', 'jpeg')):  # Process only image files
            img_path = os.path.join(blur_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

                start = time.time()
                result_image = model(img_tensor)
                stop = time.time()
                
                # Logging time for each image
                print(f'Image: {iteration}, CNN Runtime: {stop - start:.4f}')
                test_time += (stop - start)
                print(f'Average Runtime: {test_time / float(iteration):.4f}')
                
                # Post-process and save the result image
                result_image = result_image + 0.5
                out_file_name = os.path.join(out_path, img_name)
                torchvision.utils.save_image(result_image, out_file_name)
