import pydicom

class StudyCaseDICOM():
    def __init__(self, study_path):
        self.study_path = study_path

    def get_protocol_folder(self, protocol):
        if protocol == "magnitude":
            return "series0900-unknown"
        if protocol == "LR":
            return "series0901-unknown"
        if protocol == "AP":
            return "series0902-unknown"
        if protocol == "SI":
            return "series0903-unknown"
        
    def get_header(self, protocol, file_index):
        """
        file_index: index of the file (ranging from 1 to 7200, there are 7200 .dcm files in each protocol folder)
        """
        if file_index < 10:
            file_index = "000" + str(file_index)
        elif file_index >= 10 and file_index < 100:
            file_index = "00" + str(file_index)
        elif file_index >= 100 and file_index < 1000:
            file_index = "0" + str(file_index)
        elif file_index >= 1000: # don't do anything
            file_index = str(file_index)

        protocol_folder = self.get_protocol_folder(protocol)
        file_path = self.study_path + "/" + protocol_folder + "/" + "img" + file_index + "--85.6149.dcm"
        hdr = pydicom.dcmread(file_path) # loading the DICOM file

        return hdr

    def get_venc(self, protocol):
        venc_tag = (0x0019, 0x10CC) # Tag of the VENC value (where we go get it, found after looking through the entire tag list)
        hdr = self.get_header(protocol, 1)
        if venc_tag in hdr:
            venc = hdr[venc_tag].value # value of the venc in cm/s
            #print(f"VENC found (tag {venc_tag}): {hdr[venc_tag].value} cm/s")
        return venc

    def get_velocity_encode_scale(self, protocol):
        venc_scale_tag = (0x0019, 0x10E2) # Tag of the Velocity Encoding Scale (not used here, but could be useful)
        # So we can now compute the velocity by going from phase value to velocity value : velocity = voxel_value * venc_scale

        hdr = self.get_header(protocol, 1)
        if venc_scale_tag in hdr:
            venc_scale = hdr[venc_scale_tag].value
            print(f"Velocity Encoding Scale found (tag {venc_scale_tag}): {hdr[venc_scale_tag].value}cm/s/rad") # venc_scale = venc / pi
        return venc_scale
    
study_case_dicom = StudyCaseDICOM("Test/input/IRM_BAO_069_1_4D")
print(study_case_dicom.get_header("magnitude", 1))