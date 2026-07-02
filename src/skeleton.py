# skeleton.py
from pathlib import Path

import numpy as np
import smplx
import torch

from src.config import DEVICE, get_device


class Skeleton:
    def __init__(self, batch_size=1, device=None):
        self.batch_size = batch_size
        self.device = get_device(device) if device is not None else DEVICE

        model_path = Path(__file__).resolve().parent / "smplx" / "SMPLX_NEUTRAL.npz"
        self.model = smplx.create(
            model_path=str(model_path),
            model_type="smplx",
            gender="neutral",
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
            batch_size=batch_size,
        ).to(self.device)

        self.model.eval()

    def _to_tensor(self, arr, repeat=1):
        """Конвертирует numpy массив в тензор на нужном устройстве"""
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if repeat > 1:
            t = t.unsqueeze(0).repeat(repeat, 1)
        else:
            t = t.unsqueeze(0)
        return t

    def forward(self, body_pose, betas, transl, global_orient):
        with torch.inference_mode():
            output = self.model(
                body_pose=self._to_tensor(body_pose),
                betas=self._to_tensor(betas),
                transl=self._to_tensor(transl),
                global_orient=self._to_tensor(global_orient),
                return_verts=True,
            )
        vertices = output.vertices.squeeze(0).cpu().numpy()
        joints = output.joints.squeeze(0).cpu().numpy()
        return vertices, joints

    def forward_batch(self, individuals):
        """Батчевый forward для всей популяции"""
        body_pose = np.stack([ind.body_pose for ind in individuals])
        betas = np.stack([ind.betas for ind in individuals])
        transl = np.stack([ind.transl for ind in individuals])
        global_orient = np.stack([ind.global_orient for ind in individuals])

        if len(individuals) != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: expected {self.batch_size}, got {len(individuals)}"
            )

        bp = torch.tensor(body_pose, dtype=torch.float32, device=self.device)
        bt = torch.tensor(betas, dtype=torch.float32, device=self.device)
        tr = torch.tensor(transl, dtype=torch.float32, device=self.device)
        go = torch.tensor(global_orient, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            output = self.model(
                body_pose=bp,
                betas=bt,
                transl=tr,
                global_orient=go,
                return_verts=True,
            )

        joints = output.joints.cpu().numpy()
        return joints
