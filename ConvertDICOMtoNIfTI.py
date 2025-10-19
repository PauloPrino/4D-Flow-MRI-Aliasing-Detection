import subprocess
import json
import pydicom
import os

class ConvertDICOMtoNIfTI():
    def __init__(self, dicom_folder, nifti_folder):
        self.dicom_folder = dicom_folder
        self.nifti_folder = nifti_folder
        self.convert()
        self.save_dicom_header()

    def convert(self):
        for patient_folder in os.listdir(self.dicom_folder):
            os.makedirs(os.path.join(self.nifti_folder, patient_folder + "_NIfTI"), exist_ok=True)
            command = [
                "dcm2niix",
                "-o", os.path.join(self.nifti_folder, patient_folder + "_NIfTI"),
                "-f", patient_folder + "_NIfTI", # the output name preference
                "-z", "y", # compress it as a .nii.gz
                "-m", "y", # have a json with it
                os.path.join(self.dicom_folder, patient_folder)
            ]

            # Execute the command
            try:
                print(f"Using the command dcm2niiX: {command}")
                subprocess.run(command, check=True)
                print(f"Conversion finished. The NIfTI files (.nii.gz) with their json files are in the  folder {self.nifti_folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error in the conversion process for folder {patient_folder}: {e}")
            except FileNotFoundError:
                print("Error : dcm2niix is not installed or is not in the PATH of the computer.")

    def save_dicom_header(self):
        """
        Saves the dicom header content in a json file
        """
        print("Saving dicom header as json...")
        for patient_folder in os.listdir(self.dicom_folder): # go through all the patient folders
            print(f"patient folder{patient_folder}")
            for protocol in os.listdir(os.path.join(self.dicom_folder, patient_folder)): # go through all the protocols (anatomy and the 3 flow directions)
                dicom_file = os.path.join(self.dicom_folder, os.path.join(patient_folder, os.path.join(protocol, os.listdir(os.path.join(self.dicom_folder, os.path.join(patient_folder, protocol)))[0]))) # we take the first dicom file as they all have the same header
                print(f"Dicom file {dicom_file}")
                try:
                    dicom_header = pydicom.dcmread(dicom_file)
                    header_dict = {}
                    for elem in dicom_header:
                        header_dict[elem.name] = {
                            "tag": str(elem.tag),
                            "value": str(elem.value),
                            "VR": elem.VR,
                        }

                    # Save the header in a json file
                    json_name = ""
                    protocol_name = dicom_header[(0x0008,0x103E)].value
                    if protocol_name == "SAG 4DFLOW postCE - LR Flow":
                        json_name = patient_folder + "_NIfTI_e2_dicom_header.json"
                    if protocol_name == "SAG 4DFLOW postCE - AP Flow":
                        json_name = patient_folder + "_NIfTI_e3_dicom_header.json"
                    if protocol_name == "SAG 4DFLOW postCE - SI Flow":
                        json_name = patient_folder + "_NIfTI_e4_dicom_header.json"
                    if protocol_name == "SAG 4DFLOW postCE - Anatomy":
                        json_name = patient_folder + "_NIfTI_dicom_header.json"
                    json_file = os.path.join(os.path.join(self.nifti_folder, patient_folder + "_NIfTI"),json_name)

                    with open(json_file, 'w') as f:
                        json.dump(header_dict, f, indent=4, default=str)

                    print(f"Saved DICOM header for {dicom_file} to {json_file}")

                except Exception as e:
                    print(f"Error reading {dicom_file}: {e}")

ConvertDICOMtoNIfTI("Test/input", "Dataset/RawData")