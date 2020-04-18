import sys
import os
import argparse
import tqdm
from skimage import transform as tf
try:
    from .rigid_registration import load_params
except:
    from rigid_registration import load_params
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Finalize registration.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help="Input image directory",metavar='<input>')
    parser.add_argument('--first', help="First transformation parameters directory",metavar='<first>',default='fine_params')
    parser.add_argument('--nonrigid', help="Output directory of inverted non-rigid registration",metavar='<name>',default='inverted_nonrigid_output')
    parser.add_argument('--second', help="Second transformation parameters directory",metavar='<second>',default='re_rigid_params')
    parser.add_argument('--output', help="Output directory",metavar='<second>',default='final_output')
    parser.add_argument('--cval', help="cval for skimage.transform.warp",metavar='<value>',default=255,type=int)

    args = parser.parse_args()
    filenames = os.listdir(args.input)
    moving = np.array(Image.open(os.path.join(args.input,filenames[0])))
    output_shape = (np.max(moving.shape[:2]),)*2
    pad = (np.array(output_shape)-np.array(moving.shape[:2]))//2
    pad = np.append(pad,0)
    pad = np.repeat(np.expand_dims(pad,1),2,axis=1)
    padded = np.pad(moving,pad,'constant',constant_values=args.cval)
    Image.fromarray(padded).save(os.path.join(args.output,os.path.splitext(filenames[0])[0]+'.png')) #pad first image and save
    t = tf.EuclideanTransform(translation=-(np.array(output_shape)-np.array(moving.shape[:2][::-1]))//2)
    for filename in tqdm.tqdm(filenames[1:]):
        png_filename = os.path.splitext(filename)[0] + '.png'
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(os.path.join(args.second, txt_filename)):
            params = load_params(os.path.join(args.second, txt_filename))
        else:
            params = load_params(os.path.join(args.first, txt_filename))
        if os.path.exists(os.path.join(args.nonrigid, png_filename)):
            moving = np.array(Image.open(os.path.join(args.nonrigid,png_filename)))
        else:
            moving = np.array(Image.open(os.path.join(args.input,filename)))
        center = np.array(moving.shape[:2][::-1])/2-.5
        t = t + (tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(rotation=params[0],translation=params[1:]) + tf.EuclideanTransform(translation=center))
        moved = tf.warp(moving,t,order=1,preserve_range=True, output_shape=output_shape,cval=args.cval)
        Image.fromarray(np.round(moved).astype(np.uint8)).save(os.path.join(args.output,png_filename))
    return 0

if __name__ == "__main__":
    sys.exit(main())
