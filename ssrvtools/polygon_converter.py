import os
import sys
import argparse
import vtk
import tqdm

def main():
    parser = argparse.ArgumentParser(description='Convert label image into polygon mesh files.')
    parser.add_argument('input', help='Input filename',metavar='<input>')
    parser.add_argument('-o','--output', help='Output filename. Optional. Defaulted to input filename.',metavar='<output>')
    parser.add_argument('--ext', help='File extension. Default:%(default)s',metavar='<extension>',default='vtk')
    parser.add_argument('--reduce', help='Target reduction rate. Default:%(default)s',metavar='<rate>',default=0.9,type=float)
    parser.add_argument('--smooth', help='# of iteration for smoothing. Default:%(default)s',metavar='<n>',default=10,type=int)
    parser.add_argument('--cap', help="Cap on the image border",action='store_true')


    args = parser.parse_args()

    if args.output is None:
        output_format = os.path.splitext(os.path.basename(args.input))[0] + '_{}.' + args.ext
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(args.input)
    reader.Update()
    n_labels = int(reader.GetOutput().GetScalarRange()[1])

    contour = vtk.vtkDiscreteMarchingCubes()
    if args.cap:
        padder = vtk.vtkImageConstantPad()
        padder.SetInputConnection(reader.GetOutputPort())
        padder.SetConstant(0)
        extent = reader.GetOutput().GetExtent()
        padder.SetOutputWholeExtent(-1, extent[1]+1, -1, extent[3]+1, -1, extent[5]+1)
        contour.SetInputConnection(padder.GetOutputPort())
    else:
        contour.SetInputConnection(reader.GetOutputPort())
    contour.ComputeNormalsOff()

    writer = {'vtk':vtk.vtkPolyDataWriter,
              'ply':vtk.vtkPLYWriter,
              'stl':vtk.vtkSTLWriter
    }.get(args.ext, lambda : (print('Unknown file type:'+args.ext), sys.exit(1)))()
    writer.SetFileTypeToBinary()

    for index in tqdm.trange(1, n_labels + 1):
        contour.SetValue(0, index)
        contour.Update()

        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(contour.GetOutputPort())
        decimate.SetTargetReduction(args.reduce)
        decimate.Update()

        smoother= vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(decimate.GetOutputPort())
        smoother.SetNumberOfIterations(args.smooth)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        normals= vtk.vtkPolyDataNormals()
        normals.SetInputData(smoother.GetOutput())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()


        writer.SetFileName(output_format.format(index))
        writer.SetInputData(normals.GetOutput())
        writer.Write()

if __name__ == "__main__":
    main()
