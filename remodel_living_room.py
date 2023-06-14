# Futures
from __future__ import unicode_literals
from __future__ import print_function

import argparse
from pathlib import Path
# Custom imports from controlnet class
from controlnet.remodeller import ControlNetMLSD, ControlNetSegment
from utils.utils import create_dir_if_not_exists

def main(args):
    prompt = args.prompt
    img_path = args.image_path

    mlsd_net_seg = ControlNetMLSD(
        prompt=prompt, 
        image_path=img_path
    )

    img_path_parent = Path(img_path).parent
    create_dir_if_not_exists(img_path_parent+"/images")
    
    mlsd_net_seg.generate_mlsd_image(
        mlsd_save_path=f'{img_path_parent}/images/house_mlsd_{prompt.strip().replace(" ", "")}.jpeg',
        mlsd_diff_gen_save_path=f'{img_path_parent}/images/house_mlsd_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )

    control_net_seg = ControlNetSegment(
        prompt=prompt,
        image_path=img_path)
    
    seg_image = control_net_seg.segment_generation(
        save_segmentation_path=f'{img_path_parent}/images/house_seg_{prompt.strip().replace(" ", "")}.jpeg',
        save_gen_path=f'{img_path_parent}/images/house_seg_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for the image.')
    parser.add_argument('--image_path', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()
    main(args)
