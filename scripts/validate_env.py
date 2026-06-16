import os
import torch
import monai
import nnunet
import nibabel
import pydicom
import SimpleITK
import scipy
import skimage
import sklearn
from nnunet_mednext import create_mednext_v1

print("=== ENV ===")
print("nnUNet_raw_data_base:", os.environ.get("nnUNet_raw_data_base"))
print("nnUNet_preprocessed:", os.environ.get("nnUNet_preprocessed"))
print("RESULTS_FOLDER:", os.environ.get("RESULTS_FOLDER"))

print("\n=== LIBS ===")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("monai:", monai.__version__)
print("nnunet: OK")
print("nibabel:", nibabel.__version__)
print("pydicom:", pydicom.__version__)
print("SimpleITK: OK")
print("scipy:", scipy.__version__)
print("skimage:", skimage.__version__)
print("sklearn:", sklearn.__version__)

print("\n=== MEDNEXT ===")
model = create_mednext_v1(1, 2, "S", 3, False).cuda()
x = torch.randn(1, 1, 64, 64, 64).cuda()

with torch.no_grad():
    y = model(x)

print("MedNeXt OK")
print("output shape:", y.shape)
