# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import os
import glob

from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
from tqdm import tqdm


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    opt['dist'] = False
    model = create_model(opt)

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    file_client = FileClient('disk')

    input_list = sorted(glob.glob(os.path.join(img_path, '*')))
    for input_path in tqdm(input_list):
        img_bytes = file_client.get(input_path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(input_path))
        img = img2tensor(img, bgr2rgb=True, float32=True)

        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])

        destination = os.path.join(output_path, os.path.basename(input_path))
        imwrite(sr_img, destination)

if __name__ == '__main__':
    main()

