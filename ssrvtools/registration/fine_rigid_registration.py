import sys
import os
import argparse
import subprocess
import tqdm

def main():
    parser = argparse.ArgumentParser(description='Apply fine rigid registration to a series of images.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help="Input directory",metavar='<input>')
    parser.add_argument('-o','--output', help="Output directory for transformation parameters",metavar='<output>',default='fine_params')
    parser.add_argument('--cmp', help="Output directory for comparison images",metavar='<name>',default='fine_cmp')
    parser.add_argument('--input_params', help="Input directory for transformation parameters",metavar='<name>',default='coarse_params')
    parser.add_argument('--mi', help="Use mutual information",action='store_true')

    args = parser.parse_args()
    filenames = os.listdir(args.input)
    for i in tqdm.tqdm(range(1,len(filenames))):

        subprocess.check_output('fine_rigid_registration -f {} -m {} --cmp {} -i {} -p {} {}'.format(
            os.path.join(args.input,filenames[i-1]),
            os.path.join(args.input,filenames[i]),
            os.path.join(args.cmp,os.path.splitext(filenames[i])[0]+'.jpg'),
            os.path.join(args.input_params,os.path.splitext(filenames[i])[0]+'.txt'),
            os.path.join(args.output,os.path.splitext(filenames[i])[0]+'.txt'),
            '--mi' if args.mi else ''
            ), shell=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
