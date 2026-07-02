# check_device.py
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from src.config import DEVICE
print("Using device:", DEVICE)

from src.skeleton import Skeleton
import numpy as np

print("Loading SMPL-X...")
smpl = Skeleton(batch_size=1)

vertices, joints = smpl.forward(
    body_pose=np.zeros(63),
    betas=np.zeros(10),
    transl=np.zeros(3),
    global_orient=np.zeros(3),
)

print("Vertices shape:", vertices.shape)
print("Joints shape:  ", joints.shape)
print("All good!")