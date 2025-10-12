import nibabel as nib

nii = nib.load("Dataset/IRM_BAO_069_1_4D_NIfTI/IRM_BAO_069_1_4D_NIfTI_e3.nii.gz")

hdr = nii.header
print(hdr)