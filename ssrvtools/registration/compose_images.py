import sys
import argparse
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Compose images.')
    parser.add_argument('top', help="Top image",metavar='<input>')
    parser.add_argument('bottom', help="Bottom image",metavar='<param>')
    parser.add_argument('output', help="Output image",metavar='<output>')
    args = parser.parse_args()

    top = np.array(Image.open(args.top)).astype(np.float32)
    if top.shape[-1] != 4:
        print('{} has no alpha channel!'.format(args.top))
        return 1

    bottom = np.array(Image.open(args.bottom)).astype(np.float32)
    alpha = top[:,:,-1][:,:,None] / 255

    composed = alpha * top[:,:,:3] + (1-alpha) * bottom[:,:,:3]
    Image.fromarray(np.round(composed).astype(np.uint8)).save(args.output)
    return 0

if __name__ == "__main__":
    sys.exit(main())
