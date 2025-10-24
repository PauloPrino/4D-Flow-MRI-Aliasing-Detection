# 4D-Flow-MRI-Aliasing-Detection
Deep Learning for aliasing detection in 4D flow MRI.

In this repository you will find all the necessary code for 4D MRI visualizations through python (matplotlib). You will also find scripts to simulate the aliasing effect on real world 4D MRI data.

## Converting data form DICOM to NIfTI

The output data type of an MRI machine is commonly the DICOM (.dcm) file type. However, this file type takes a lot of memory space which makes it computationally less effective and difficult to manipulate while working with large amounts of data in the case of deep learning. This is why before any work is done one the data, we convert these DICOM files into NIfTI ones. More precisely we convert them to .nii.gz files as it is a compression of the classic NIfTI files (.nii) by compressing the background. Because **dcm2nixx** does not keep all of the information of the DICOM header into the NIfTI header we also extract the header of the DICOMs by hand and put them with the NIfTIs outputed to keep important data such as the velocity encoding which is not in the NIfTI header.
To convert the DICOM files to .nii.gz files we use the converter dcm2niix that you can download here: https://github.com/rordenlab/dcm2niix/releases depending on your setup. The file **ConvertDICOMtoNIfTI.py** is in charge of this conversion. It is based on a function that you need to feed with the *input folder* where your DICOM folders are stored and the *output folder* where you want the .nii.gz files to be put.

## Work on NIfTI files with Python

To work with your NIfTI files with Python you can use the script **StudyCaseNIfTI.py** which is a class based script with several methods to play with the nifti data: visualization, getters, setters etc.

![](ReadmeResources\Coronnal_Slice_166_Anatomy.png)
Anatomy visualization of a coronal slice
![](Results\Animations\velocity_sagittal_slice_70_with_magnitude_mask.gif)
Visualization of blood flow with vectors on a sagittal slice