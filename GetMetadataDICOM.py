import pydicom

ds = pydicom.dcmread("Dataset/IRM_BAO_069_1_4D/series0901-unknown/img0001--85.6149.dcm") # loading the DICOM file

for elem in ds: # listing all tags of the document
    print(elem) # Tag name, value representation, value length, and value


venc_tag = (0x0019, 0x10CC) # Tag of the VENC value (where we go get it, found after looking through the entire tag list)
velocity_encode_scale_tag = (0x0019, 0x10E2) # Tag of the Velocity Encoding Scale (not used here, but could be useful)
# So we can now compute the velocity by going from phase value to velocity value : velocity = voxel_value / velocity_encode_scale

if venc_tag in ds:
    print(f"VENC found (tag {venc_tag}): {ds[venc_tag].value} mm/s so {ds[venc_tag].value/10} cm/s")
if velocity_encode_scale_tag in ds:
    print(f"Velocity Encoding Scale found (tag {velocity_encode_scale_tag}): {ds[velocity_encode_scale_tag].value}")