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

        for elem in hdr: # listing all tags of the document
            print(elem) # Tag name, value representation, value length, and value
        return hdr

    def get_venc(self, protocol):
        venc_tag = (0x0019, 0x10CC) # Tag of the VENC value (where we go get it, found after looking through the entire tag list)
        hdr = self.get_header(protocol, 1)
        if venc_tag in hdr:
            venc = hdr[venc_tag].value # value of the venc in mm/s
            print(f"VENC found (tag {venc_tag}): {hdr[venc_tag].value} mm/s so {hdr[venc_tag].value/10} cm/s")
        return venc

    def get_velocity_encode_scale(self, protocol):
        velocity_encode_scale_tag = (0x0019, 0x10E2) # Tag of the Velocity Encoding Scale (not used here, but could be useful)
        # So we can now compute the velocity by going from phase value to velocity value : velocity = voxel_value / velocity_encode_scale

        hdr = self.get_header(protocol, 1)
        if velocity_encode_scale_tag in hdr:
            velocity_encode_scale = hdr[velocity_encode_scale_tag].value
            print(f"Velocity Encoding Scale found (tag {velocity_encode_scale_tag}): {hdr[velocity_encode_scale_tag].value}s/mm")
        return velocity_encode_scale
    
study_case_dicom = StudyCaseDICOM("Dataset/IRM_BAO_069_1_4D")