import sys
import argparse
import SimpleITK as sitk
from math import pi
import numpy as np
from PIL import Image
import time

def load_image(filename):
    rgb = np.array(Image.open(filename))
    if rgb.shape[-1] == 3:
        return np.concatenate((rgb,
                           255*np.ones_like(rgb[:,:,:1],dtype=np.uint8)),axis=-1)
    else:
        return rgb

def load_params(filename):
    with open(filename, 'r') as f:
        return [float(e) for e in f.readline().split()]

def write_params(filename, params):
    with open(filename, 'w') as f:
        f.write(' '.join([str(e) for e in params]))

def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))

def fine_registration():
    parser = argparse.ArgumentParser(description='Register moving image to fixed image.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--fixed', help="Fixed image filename",metavar='<filename>',required=True)
    parser.add_argument('-m','--moving', help="Moving image filename",metavar='<filename>',required=True)
    parser.add_argument('-i','--init_params', help="Initial parameter",metavar='<filename>')
    parser.add_argument('--iter', help="Max number of iterations",metavar='<n>',default=16,type=int)
    parser.add_argument('-o','--output', help="Output moved image filename",metavar='<filename>')
    parser.add_argument('-p','--params', help="Output transformation parameter",metavar='<filename>')
    parser.add_argument('-c','--cmp', help="Output comparison image filename",metavar='<filename>')
    parser.add_argument('--mi', help="Use mutual information as the similarity metric in stead of MeanSquares",action='store_true')
    parser.add_argument('-v','--verbose', help="Activate verbose mode",action='store_true')
    parser.add_argument('--factor', help="Shrinking factor for multi level optimization",metavar='<rate>',default=[10,1],type=int,nargs='+')
    parser.add_argument('--rate', help="Sampling rate for the metric computation",metavar='<rate>',default=[1,.1],type=float,nargs='+')
    parser.add_argument('--random', help="Use RANDOM sampling strategy instead of REGULAR",action='store_true')
    parser.add_argument('--sigma', help="Sigma for gaussian smoothing",metavar='<value>',default=[5,5],type=float,nargs='+')

    args = parser.parse_args()

    fixed_rgba = load_image(args.fixed)
    moving_rgba = load_image(args.moving)
    fixed_gray = fixed_rgba[:,:,0]
    fixed = sitk.GetImageFromArray(fixed_gray.astype(np.float32))
    # fixed = sitk.DiscreteGaussian( fixed, args.sigma )
    moving = sitk.GetImageFromArray(moving_rgba[:,:,0].astype(np.float32))
    # moving = sitk.DiscreteGaussian(moving, args.sigma)

    R = sitk.ImageRegistrationMethod()
    if args.mi:
        R.SetMetricAsMattesMutualInformation(64)
    else:
        R.SetMetricAsMeanSquares()

    tx = sitk.Euler2DTransform()
    tx.SetCenter(np.array(fixed_gray.shape[::-1])/2-.5)
    R.SetOptimizerAsPowell(args.iter, stepLength=0.1)
    R.SetMetricSamplingStrategy(R.RANDOM if args.random else R.REGULAR)
    R.SetMetricSamplingPercentagePerLevel(args.rate)
    R.SetShrinkFactorsPerLevel(args.factor)
    R.SetSmoothingSigmasPerLevel(args.sigma)

    if args.init_params:
        tx.SetParameters(load_params(args.init_params))
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkNearestNeighbor)

    if args.verbose:
        R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    tic = time.time()
    outTx = R.Execute(fixed, moving)
    print(time.time() - tic)
    params = np.array(outTx.GetParameters())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(fixed_gray))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(255)
    resampler.SetTransform(outTx)

    out = resampler.Execute(sitk.GetImageFromArray(moving_rgba[:,:,0]))
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    if args.output:
        rgba = []
        for i in range(moving_rgba.shape[-1]):
            out = resampler.Execute(sitk.GetImageFromArray(moving_rgba[:,:,i]))
            rgba.append(sitk.GetArrayFromImage(sitk.Cast(out, sitk.sitkUInt8)))
            # Image.fromarray(sitk.GetArrayFromImage(simg2)).save(args.output)
        Image.fromarray(np.stack(rgba,axis=-1)).save(args.output)
    if args.cmp:
        simg1 = sitk.Cast(sitk.RescaleIntensity(sitk.GetImageFromArray(fixed_gray)), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
        Image.fromarray(sitk.GetArrayFromImage(cimg)).save(args.cmp)

    if args.params:
        write_params(args.params, params)

    return 0

def coarse_registration():
    parser = argparse.ArgumentParser(description='Register moving image to fixed image.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--fixed', help="Fixed image",metavar='<filename>',required=True)
    parser.add_argument('-m','--moving', help="Moving image",metavar='<filename>',required=True)
    parser.add_argument('-o','--output', help="Output moved image",metavar='<filename>')
    parser.add_argument('-p','--params', help="Output parameters",metavar='<filename>')
    parser.add_argument('-c','--cmp', help="Output comparison image",metavar='<filename>')
    parser.add_argument('--mi', help="Use mutual information as the metric in stead of MeanSquares",action='store_true')
    parser.add_argument('-s','--scale', help="Scaling factor",metavar='<n>',default=30,type=int)
    parser.add_argument('-t','--tmax', help="Maximum translation",metavar='<n>',type=int)
    parser.add_argument('--dt', help="Step size for translation",metavar='<n>',type=int,default=2)
    parser.add_argument('--n_angle_sample', help="Sample size for angle",metavar='<n>',type=int,default=128)
    parser.add_argument('--sigma', help="Sigma for gaussian smoothing",metavar='<value>',default=2,type=float)

    args = parser.parse_args()

    fixed_rgba = load_image(args.fixed)
    moving_rgba = load_image(args.moving)
    fixed_gray = fixed_rgba[:,:,0]
    scale = args.scale
    fixed = sitk.GetImageFromArray(fixed_gray[::scale,::scale].astype(np.float32))
    fixed = sitk.DiscreteGaussian( fixed, args.sigma );
    moving = sitk.GetImageFromArray(moving_rgba[:,:,0][::scale,::scale].astype(np.float32))
    moving = sitk.DiscreteGaussian(moving, args.sigma)

    R = sitk.ImageRegistrationMethod()
    if args.mi:
        R.SetMetricAsMattesMutualInformation(64)
    else:
        R.SetMetricAsMeanSquares()

    sample_per_axis=args.n_angle_sample
    tx = sitk.Euler2DTransform()
    if args.tmax is None:
        tmax = int(min(fixed_gray.shape)/scale//5)
    else:
        tmax = int(args.tmax // scale)
    dt = args.dt
    tx.SetCenter(np.array(fixed_gray.shape[::-1])/scale/2-.5)
    R.SetOptimizerAsExhaustive([sample_per_axis//2,tmax//2,tmax//2])
    R.SetOptimizerScales([2.0*pi/sample_per_axis, dt, dt])

    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    tic = time.time()
    outTx = R.Execute(fixed, moving)
    outTx.SetFixedParameters(np.array(fixed_gray.shape[::-1])/2-.5)
    print(time.time() - tic)
    # scale parameters
    params = np.array(outTx.GetParameters())
    params[1] *= scale
    params[2] *= scale
    outTx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(fixed_gray))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(sitk.GetImageFromArray(moving_rgba[:,:,0]))

    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    if args.output:
        Image.fromarray(sitk.GetArrayFromImage(simg2)).save(args.output)
    if args.cmp:
        simg1 = sitk.Cast(sitk.RescaleIntensity(sitk.GetImageFromArray(fixed_gray)), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
        Image.fromarray(sitk.GetArrayFromImage(cimg)).save(args.cmp)

    if args.params:
        write_params(args.params, params)

    return 0

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage:',sys.argv[0],'<coarse|fine>')
    else:
        command = sys.argv[1]
        del sys.argv[1]
        if command == 'fine':
            sys.exit(fine_registration())
        else:
            sys.exit(coarse_registration())
