import argparse
import mhd
from scipy.ndimage.filters import median_filter
import _label_filters

def modefilt():
    parser = argparse.ArgumentParser(description='Mode filter.')
    parser.add_argument('input', help="Input mhd filename",metavar='<input>')
    parser.add_argument('-o','--output', help="Output mhd filename",metavar='<filename>',default='filtered.mha')
    parser.add_argument('--size', help="Filter size",metavar='<N>',default=3,type=int)

    args = parser.parse_args()
    label, h = mhd.read(args.input)
    filtered = _label_filters.modefilt3(label, args.size, 0)
    mhd.write(args.output, filtered, h)


def medfilt():
    parser = argparse.ArgumentParser(description='Median filter.')
    parser.add_argument('input', help="Input mhd filename",metavar='<input>')
    parser.add_argument('-o','--output', help="Output mhd filename",metavar='<filename>',default='filtered.mha')
    parser.add_argument('--size', help="Filter size",metavar='<N>',default=3,type=int)

    args = parser.parse_args()
    label, h = mhd.read(args.input)
    filtered = median_filter(label, args.size)
    mhd.write(args.output, filtered, h)

if __name__ == "__main__":
    medfilt()
