import numpy as np
from PIL import Image
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import argparse
import os

def register_rigid(fixed, moving):
    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))
    params = sitk.GetDefaultParameterMap("rigid")
    params['MaximumNumberOfIterations'] = ['32']
    params['NumberOfResolutions'] = ['1']
    params['DefaultPixelValue'] = ['255']
    # params['ImageSampler'] = ['Full']
    elastixImageFilter.SetParameterMap(params)
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.Execute()

    img = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    return img, transformParameterMap

def register_nonrigid(fixed, fixed_mask, moving, moving_mask, max_iter):
    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))
    if fixed_mask is not None:
        elastixImageFilter.SetFixedMask(sitk.GetImageFromArray(fixed_mask))
    if moving_mask is not None:
        elastixImageFilter.SetMovingMask(sitk.GetImageFromArray(moving_mask))
    params = sitk.GetDefaultParameterMap("bspline")
    params['MaximumNumberOfIterations'] = [str(max_iter)]
    params['NumberOfResolutions'] = ['1']
    params['FinalGridSpacingInPhysicalUnits'] = ['256']
    params['GridSpacingSchedule'] = ['1']
    params['DefaultPixelValue'] = ['255']
#    params['NumberOfHistogramBins'] = ['64']
    # params['NumberOfSpatialSamples'] = [str(int(fixed.size*.05))]
    # params['ImageSampler'] = ['Full']
    elastixImageFilter.SetParameterMap(params)
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.Execute()

    img = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    return img, transformParameterMap[0]

def deform(moving, params):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(params)

    transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.Execute()
    img = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
    return np.round(np.clip(img,0,255)).astype(np.uint8)

def calc_params(fixed_rgb, moving_rgb, rigid=False, max_iter=64):
    fixed = fixed_rgb[:,:,0]
    if fixed_rgb.shape[-1] in (4,2):
        fixed_mask = fixed_rgb[:,:,-1]
    else:
        fixed_mask = None
    moving = moving_rgb[:,:,0]
    if moving_rgb.shape[-1] in (4,2):
        moving_mask = moving_rgb[:,:,-1]
    else:
        moving_mask = None
    sigma = 4
    fixed = gaussian_filter(fixed,sigma=sigma)
    moving = gaussian_filter(moving,sigma=sigma)
    if rigid:
        moved,pmap = register_rigid(fixed,moving)
    else:
        moved,pmap = register_nonrigid(fixed,fixed_mask,moving,moving_mask,max_iter=max_iter)
    moved = deform(moving_rgb[:,:,0],pmap)
    moved = np.stack((moved,deform(moving_rgb[:,:,-1],pmap)),axis=-1)
    return pmap, moved

def main():
    parser = argparse.ArgumentParser(description='Non-rigid registration.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fixed1', help="Fixed image 1",metavar='<filename>',required=True)
    parser.add_argument('--fixed2', help="Fixed image 2",metavar='<filename>',required=True)
    parser.add_argument('--moving', help="Moving image",metavar='<filename>',required=True)
    parser.add_argument('--output', help="Output image",metavar='<name>',required=True)
    parser.add_argument('--iter', help="Maximum number of iterations",metavar='<n>',type=int,default=64)
    parser.add_argument('--t', help="Coefficient for linear interpolation",metavar='<coeff>',type=float,default=.5)
    parser.add_argument('--cmp', help="Output cmp image. Use '.tif' for multi-frame output.",metavar='<name>',nargs='*')
    args = parser.parse_args()

    fixed1_rgb = np.array(Image.open(args.fixed1))
    fixed2_rgb = np.array(Image.open(args.fixed2))
    moving_rgb = np.array(Image.open(args.moving))

    print('rigid1')
    pr, fixed2_moved = calc_params(fixed1_rgb,fixed2_rgb,True)
    # print('rigid2')
    # pr2, moving_moved = calc_params(fixed1_rgb,moving_rgb,True)

    print('non-rigid1')
    p1,_ = calc_params(fixed1_rgb,moving_rgb,max_iter=args.iter)
    print('non-rigid2')
    p2,_ = calc_params(fixed2_moved,moving_rgb,max_iter=args.iter)

    v1 = np.array([float(e) for e in p1['TransformParameters']])
    v2 = np.array([float(e) for e in p2['TransformParameters']])
    v = args.t * v1 + (1-args.t) * v2

    p1['TransformParameters'] = [str(e) for e in v]
    deformed = deform(moving_rgb[:,:,0],p1)
    deformed_rgb = np.stack([deform(moving_rgb[:,:,i],p1) for i in range(moving_rgb.shape[-1])],axis=-1)
    Image.fromarray(deformed_rgb).save(args.output)
    for cmp_filename in args.cmp:
        if os.path.splitext(cmp_filename)[1]=='.tif':
            Image.fromarray(fixed1_rgb[:,:,0]).save(cmp_filename,save_all=True,append_images=[Image.fromarray(deformed),Image.fromarray(fixed2_moved[:,:,0])])
        else:
            Image.fromarray(np.stack([fixed1_rgb[:,:,0],deformed,fixed2_moved[:,:,0]],axis=-1)).save(cmp_filename)

if __name__ == "__main__":
    main()
