import argparse

def median_filter():
    parser = argparse.ArgumentParser(description='Median filter for mhd image.')
    parser.add_argument('input', help="Input filename",metavar='<input>')
    parser.add_argument('-o','--output', help="Output filename. Defualt:%(default)s",metavar='<output>',default='filtered.mha')
    parser.add_argument('-s','--size', help="Optional argument. Default:%(default)s",metavar='<n>',default=3,type=int)

    args = parser.parse_args()
    import mhd
    from scipy.ndimage.filters import median_filter
    image, h = mhd.read(args.input)
    filtered =  median_filter(image, args.size)
    mhd.write(args.output,filtered,h)

if __name__ == "__main__":
    median_filter()