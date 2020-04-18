import argparse
import sys
import os
import glob
import multiprocessing
import tqdm
from PIL import Image
import numpy as np
import ssrvtools.mhd
from sklearn.neighbors import KDTree

def multiply_alpha(image):
    a = image[:,:,3] / 255.0
    return (image[:,:,:3] * np.expand_dims(a, -1)).astype(np.uint8)

class Loader(object):
    def __init__(self, stride):
        self.stride = stride
    def __call__(self, filename):
        img = np.array(Image.open(filename))[::self.stride,::self.stride]
        if img.ndim == 3 and img.shape[-1] > 3:
            return multiply_alpha(img)
        else:
            return img

def load_images(filenames,stride=1):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        images = list(tqdm.tqdm(pool.imap(Loader(stride), filenames), total=len(filenames), desc='Loading images'))
        return np.stack(images)

class LabelConberter(object):
    def __init__(self, cmap):
        self.tree = KDTree(cmap)
    def __call__(self, image):
        rgb_points = np.reshape(image,(-1,3))
        _, converted = self.tree.query(rgb_points)
        return converted

def to_label(image, cmap):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        converted_images = list(tqdm.tqdm(pool.imap(LabelConberter(cmap), image), total=len(image), desc='Converting'))
    converted = np.stack(converted_images)
    return np.reshape(converted.astype(np.uint8), image.shape[:-1])

def process_input(args):
    filenames = None
    if len(args.input) == 1:
        if os.path.isdir(args.input[0]):
            filenames = glob.glob(os.path.join(args.input[0],'*'+args.ext))
    if not filenames:
        filenames = args.input
    return sorted(filenames)


def convert(is_label):
    parser = argparse.ArgumentParser(description='Convert image series into one mhd image.')
    parser.add_argument('input', help='Input directory',metavar='<input>',nargs='+')
    parser.add_argument('-o','--output', help='Output filename. Default:%(default)s',metavar='<output>',default='label.mha' if is_label else 'image.mha')
    parser.add_argument('--ext', help='File extension for image files. Default:%(default)s',metavar='<extension>',default='.png')
    parser.add_argument('--colormap', help='Colormap file for converting label image.',metavar='<filename>',required=False)
    parser.add_argument('--stride',help='Stride size for downsampling. Default:%(default)s',metavar='<n>',type=int,default=4)
    parser.add_argument('--spacing',help='Spacing (voxel size) for the output image',metavar='<mm>',type=float,default=None,nargs='*')
    parser.add_argument('--exclude',help='Pattern(s) to be excluded',metavar='<pattern>',nargs='*')
    parser.add_argument('--channel',help='Output only specified channel(s)',metavar='<n>',nargs='*')

    args = parser.parse_args()
    filenames = process_input(args)
    if args.exclude is not None:
        for e in args.exclude:
            filenames = [filename for filename in filenames if not e in filename]

    volume = load_images(filenames,args.stride)
    header = {}
    header['TransformMatrix'] = '-1 0 0 0 -1 0 0 0 1' #for 3DSlicer
    if args.channel:
        volume = volume[:,:,:,int(args.channel[0])]

    if args.spacing is not None:
        header['ElementSpacing'] = args.spacing
    if is_label:
        if args.colormap is None:
            args.colormap = os.path.join(os.path.dirname(__file__), 'colormap.csv')
        cmap = np.loadtxt(args.colormap, delimiter=',')
        volume = to_label(volume, cmap)
        header['CompressedData'] = True
    else:
        header['CompressedData'] = False
        if volume.shape[-1] == 3:
            header['ElementNumberOfChannels'] = volume.shape[-1]

    mhd.write(args.output,volume,header)

def img2mhd():
    return convert(False)

def label2mhd():
    return convert(True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:',sys.argv[0],'<command>','<args>')
        sys.exit()

    commands = ['img2mhd','label2mhd']
    if sys.argv[1] in commands:
        e = sys.argv[1]+'()'
        del sys.argv[1]
        eval(e)
    else:
        print('Error:Invalid command "{}"'.format(sys.argv[1]))
        print('Valid commands:',commands)
