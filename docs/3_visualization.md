# Surface extraction and Visualization

## Stack up labels
- Input \<indir\>: Input directory containing labeled images.
```bash
label2mhd <indir>
```
- Output: label.mha


## (Optional) apply median filter
- Input \<input\>: Input volume file. e.g. label.mha
```bash
median_filter <input>
```
- Output: filtered.mha

## Surface extraction
- Input \<input\>: Input volume file. e.g. label.mha or filtered.mha
```bash
mhd2polygon <input>
```
- Output(s): label_1.vtk (and label_2.vtk and so on ...)

## Visualization
Extracted surface files are in [VTK](https://lorensen.github.io/VTKExamples/site/VTKFileFormats/) Polygonal Data format. You can use visualization softwares such as [ParaView](https://www.paraview.org/) to load and visualize them.