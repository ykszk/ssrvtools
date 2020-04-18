import sys
import os
import argparse
import tqdm
from skimage import transform as tf
try:
    from .rigid_registration import load_params, write_params, load_image
except:
    from rigid_registration import load_params, write_params, load_image
import numpy as np
from PIL import Image

def _apply_rigid_transformation(inverse):
    parser = argparse.ArgumentParser(description='Apply {} transformation to a image.'.format('inverse rigid' if inverse else 'rigid'))
    parser.add_argument('input', help="Input image",metavar='<input>')
    parser.add_argument('params', help="Input transformation parameter",metavar='<param>')
    parser.add_argument('output', help="Output image",metavar='<output>')
    parser.add_argument('--size', help="Output image size",metavar='<pix>',type=int,nargs=2)
    args = parser.parse_args()

    params = load_params(args.params)
    moving = np.array(Image.open(args.input))
    if args.size:
        output_shape = np.array(args.size[::-1])
    else:
        output_shape = np.array((max(moving.shape[:2]),)*2)
    warped_shape = np.max(np.stack((moving.shape[:2],output_shape)),axis=0)
    center = np.array(warped_shape)/2-.5
    ds = np.array(output_shape[::-1])-np.array(moving.shape[:2][::-1])
    if inverse:
        t = tf.EuclideanTransform(translation=-np.clip(ds,0,None)//2) + tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(translation=-np.array(params[1:])) + tf.EuclideanTransform(rotation=-params[0]) + tf.EuclideanTransform(translation=center) + tf.EuclideanTransform(translation=-np.clip(-ds,0,None)//2)
    else:
        t = tf.EuclideanTransform(translation=-np.clip(ds,0,None)//2) + tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(rotation=params[0],translation=params[1:]) + tf.EuclideanTransform(translation=center)
    moved = tf.warp(moving,t,order=1,preserve_range=True,output_shape=warped_shape)
    pad = np.clip(-ds[::-1],0,None)//2
    moved = moved[pad[0]:(moved.shape[0]-pad[0]),pad[1]:(moved.shape[1]-pad[1])]
    Image.fromarray(np.round(moved).astype(np.uint8)).save(os.path.join(args.output))

def apply_rigid_transformation():
    _apply_rigid_transformation(False)

def apply_inverse_rigid_transformation():
    _apply_rigid_transformation(True)

def main():
    parser = argparse.ArgumentParser(description='Apply fine rigid registration to a series of images.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help="Input image directory",metavar='<input>')
    parser.add_argument('--input_params', help="Input directory for transformation parameters",metavar='<name>',default='fine_params')
    parser.add_argument('--composed_params', help="Output directory for composed transformation parameters (Optional)",metavar='<name>',default='composed_params')
    parser.add_argument('--output', help="Output directory",metavar='<name>',default='fine_output')
    parser.add_argument('--size', help="Output image size",metavar='<size>',type=int)
    parser.add_argument('--cval', help="cval for skimage.transform.warp",metavar='<value>',default=255,type=int)

    args = parser.parse_args()
    filenames = os.listdir(args.input)
    moving = np.array(Image.open(os.path.join(args.input,filenames[0])))
    if args.size:
        shape = (args.size, args.size)
    else:
        shape = (max(moving.shape[:2]),)*2
    t = tf.EuclideanTransform(translation=-(np.array(shape)-np.array(moving.shape[:2][::-1]))//2)
    for i in tqdm.tqdm(range(1,len(filenames))):
        params = load_params(os.path.join(args.input_params,os.path.splitext(filenames[i])[0]+'.txt'))
        moving = load_image(os.path.join(args.input,filenames[i]))
        center = np.array(moving.shape[:2][::-1])/2-.5
        t = t + (tf.EuclideanTransform(translation=-center) + tf.EuclideanTransform(rotation=params[0],translation=params[1:]) + tf.EuclideanTransform(translation=center))
        moved = tf.warp(moving[:,:,:3],t,order=1,preserve_range=True, output_shape=shape,cval=args.cval)
        alpha = tf.warp(moving[:,:,-1],t,order=1,preserve_range=True, output_shape=shape,cval=0)
        moved = np.concatenate([moved,alpha[:,:,None]],axis=-1)
        Image.fromarray(np.round(moved).astype(np.uint8)).save(os.path.join(args.output,os.path.splitext(filenames[i])[0]+'.png'))
        if args.composed_params:
            tt = tf.EuclideanTransform(translation=center) + t + tf.EuclideanTransform(translation=-center)
            params = [tt.rotation, tt.translation[0], tt.translation[1]]
            write_params(os.path.join(args.composed_params,os.path.splitext(filenames[i])[0]+'.txt'), params)

if __name__ == "__main__":
    sys.exit(apply_inverse_rigid_transformation())
