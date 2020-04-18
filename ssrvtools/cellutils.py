import argparse
import sys, os
import re
from PIL import Image
import numpy as np
import ssrvtools.boundingbox as bb
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from scipy import ndimage
import tqdm
import scipy.spatial
import mhd
import multiprocessing

class Searcher(object):
    def __init__(self, tree):
        self.tree = tree
    def __call__(self, pts):
        d, i = self.tree.query(pts)
        return i

def _load_image(filename):
    image = np.array(Image.open(filename))
    if image.shape[-1] == 4:
        alpha = image[:,:,-1]/255
        image = np.clip(np.round(image[:,:,0]*alpha+255*(1-alpha)),0,255).astype(np.uint8)
    elif image.ndim > 2:
        image = image[:,:,1]
    else:
        pass
    return image

def _label_cells(wall):
    labels, n_labels = ndimage.label(wall==0)
    count = np.bincount(labels.flatten())
    count_image = count[labels]
    labels[(count_image > (count_image.size/100)) & np.logical_not(wall)]  = 0 #bg
    label_contour = (ndimage.binary_dilation(wall,structure=np.ones((3,3))) ^ wall)
    min_area = 20
    # label_contour[(count_image > (count_image.size/100))]  = 0 #bg
    label_contour[count_image < min_area]  = 0

    cell_pts = np.array(np.where(label_contour)).T
    cell_tree = scipy.spatial.KDTree(cell_pts)
    wall_pts = np.array(np.where(wall)).T
    wall_label = np.zeros_like(labels)

    ps = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=ps) as pool:
        ref_ids = list(tqdm.tqdm(pool.imap(Searcher(cell_tree), np.array_split(wall_pts,ps*3)), total=ps*3, desc='Filling gaps'))
    ref_ids = np.concatenate(ref_ids,axis=0)
    for pts, ref_id in zip(wall_pts,ref_ids):
        ref_pts = cell_pts[ref_id]
        wall_label[pts[0],pts[1]] = labels[ref_pts[0],ref_pts[1]]

    labels[wall_label>0] = wall_label[wall_label>0]
    return labels

def _assign_labels(cells, ref_labels):
    new_label = np.zeros_like(ref_labels[0])
    for v in tqdm.trange(1,np.max(cells)+1,desc='Labeling'):
        m = cells==v
        count = np.bincount(ref_labels[m[None,:,:] & (ref_labels!=0)])
        if (len(count)==0):
            continue
        l = np.argmax(count)
        new_label[cells==v] = l
    return new_label

def remove_small_area(binary, min_area=500):
    label_im, nb_labels = ndimage.label(binary)
    freq = np.bincount(label_im.flatten())
    label_im[label_im==np.argmax(freq)] = 0
    removing_areas = np.where(freq<min_area)[0]
    label_im[np.isin(label_im,removing_areas)] = 0
    return label_im > 0

def _local_thresh(gray, block_size, offset):
    local_thresh = threshold_local(gray, block_size, offset=offset, method='gaussian')
    wall = (gray < local_thresh)
    return wall

def dilate(image):
    from scipy import ndimage
    return ndimage.binary_dilation(image,structure=np.ones((3,3)))

def color_cells(labels):
    vals = np.linspace(0,255,np.max(labels)+1).astype(np.int32)
    np.random.shuffle(vals)
    cm = np.array([plt.cm.gist_ncar(e) for e in vals])
    rgba = cm[labels]
    rgba =  np.clip(np.round(rgba*255),0,255).astype(np.uint8)
    rgba = rgba * (labels!=0)[:,:,None]
    return rgba

def _extract_wall(image,block_size,offset):
    original_shape = image.shape
    bbox = bb.bbox(image)
    image = bb.crop(image,bbox,margin=1)
    wall = remove_small_area(_local_thresh(image,block_size,offset))
    from scipy.ndimage.morphology import binary_closing
    wall = binary_closing(wall)
    wall = bb.uncrop(wall, original_shape, bbox, margin=1).astype(np.uint8)
    return wall

_default_block_size = 21
_default_offset = 1
def extract_wall():
    parser = argparse.ArgumentParser(description='Extract cell wall.')
    parser.add_argument('input', help="Input image filename",metavar='<input>')
    parser.add_argument('output', help="Output image filename",metavar='<output>')
    parser.add_argument('-b','--block_size', help="Block size for local threshold (default:{})".format(_default_block_size),default=_default_block_size,type=int,metavar='<size>')
    parser.add_argument('--offset', help="Offset for local threshold (default:{})".format(_default_offset),default=_default_offset,type=int,metavar='<n>')

    args = parser.parse_args()
    image = _load_image(args.input)
    wall = _extract_wall(image, args.block_size, args.offset)
    if os.path.splitext(args.output) in ['.mhd','.mha']:
        mhd.write(args.output, wall.astype(np.uint8))
    else:
        cmap = np.array([[0,0,0,0],[255,0,0,255]]).astype(np.uint8)
        Image.fromarray(cmap[wall]).save(args.output)

def label_cells():
    parser = argparse.ArgumentParser(description='Label cells.')
    parser.add_argument('input', help="Input image filename",metavar='<input>')
    parser.add_argument('output', help="Output image filename",metavar='<output>')
    parser.add_argument('-b','--block_size', help="Block size for local threshold (default:{})".format(_default_block_size),default=_default_block_size,type=int,metavar='<size>')
    parser.add_argument('--offset', help="Offset for local threshold (default:{})".format(_default_offset),default=_default_offset,type=int,metavar='<n>')

    args = parser.parse_args()
    image = _load_image(args.input)
    wall = _extract_wall(image, args.block_size, args.offset)
    cells = _label_cells(wall)
    if os.path.splitext(args.output)[1] in ['.mhd','.mha']:
        mhd.write(args.output,cells)
    else:
        rgba = color_cells(cells)
        Image.fromarray(rgba).save(args.output)

def rgba2label(rgba):
    unique_colors = np.unique(rgba[:,::8,::8].reshape((-1,rgba.shape[-1])),axis=0)
    unique_colors = np.array(sorted(unique_colors,key=lambda x:x[-1]))
    reshaped = rgba.reshape((-1,rgba.shape[-1]))
    converted = np.zeros(len(reshaped),dtype=np.uint8)
    for i in range(1,len(unique_colors)):
        c = unique_colors[i]
        m = reshaped == c[None,:]
        m = np.all(m, axis=-1)
        converted[m] = i
    return converted.reshape(rgba.shape[:-1]), unique_colors

def assign_labels():
    parser = argparse.ArgumentParser(description='Assign labels.')
    parser.add_argument('-i','--input', help="Input image filename(s)",metavar='<filename>',required=True,nargs='+')
    parser.add_argument('-r','--reference', help="Input reference image filename(s)",metavar='<filename>',required=True,nargs='+')
    parser.add_argument('-o','--output', help="Output directory (default:.)",metavar='<dirname>',default='.')
    parser.add_argument('-b','--block_size', help="Block size for local threshold (default:{})".format(_default_block_size),default=_default_block_size,type=int,metavar='<size>')
    parser.add_argument('--offset', help="Offset for local threshold (default:{})".format(_default_offset),default=_default_offset,type=int,metavar='<n>')

    args = parser.parse_args()
    refs = np.stack([np.array(Image.open(filename)) for filename in args.reference])
    refs, cmap = rgba2label(refs)
    for input_filename in args.input:
        matches = re.findall('\d+',os.path.basename(input_filename))
        if matches:
            output_filename = matches[-1]+'.png'
        else:
            output_filename = os.path.basename(input_filename)
        image = _load_image(input_filename)
        wall = _extract_wall(image, args.block_size, args.offset)
        cells = _label_cells(wall)
        new_labels = _assign_labels(cells,refs)
        rgba = cmap[new_labels]
        Image.fromarray(rgba).save(os.path.join(args.output,output_filename))

def main():
    commands = ['extract_wall','label_cells','assign_labels']
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h','--help']:
        print('Usage:'+sys.argv[0]+' <command> <args> ...')
        print('Commands:',commands)
        return 1
    if sys.argv[1] not in commands:
        print('Error: unknown command "{}"'.format(sys.argv[1]))
        print('Commands:',commands)
        return 1

    command = sys.argv[1]
    del sys.argv[0]
    return eval(command+'()')

if __name__ == "__main__":
    sys.exit(main())
