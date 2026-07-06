# skeleton.py
import torch
import numpy as np
from smplx import create


class Skeleton:

    def __init__(self, model_path="src", batch_size=50):

        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Модель для одиночного forward (batch_size=1)
        self.model_single = create(
            model_path=model_path,
            model_type="smplx",
            gender="neutral",
            batch_size=1,
            use_pca=False,
            flat_hand_mean=True
        ).to(self.device)

        # Модель для батчевого forward (batch_size=50)
        self.model_batch = create(
            model_path=model_path,
            model_type="smplx",
            gender="neutral",
            batch_size=batch_size,
            use_pca=False,
            flat_hand_mean=True
        ).to(self.device)

        print(f"Skeleton device     : {self.device}")
        print(f"Batch size          : {batch_size}")

        # Буферы для одиночного forward
        self._pose_buf   = torch.zeros(1, 63, device=self.device)
        self._betas_buf  = torch.zeros(1, 10, device=self.device)
        self._transl_buf = torch.zeros(1,  3, device=self.device)
        self._orient_buf = torch.zeros(1,  3, device=self.device)

    # ---------------------------------------------------------
    # Одиночный forward — для финального результата и отладки
    # ---------------------------------------------------------

    def forward(self, body_pose, betas, transl, global_orient):

        self._pose_buf[0]   = torch.from_numpy(body_pose.astype(np.float32))
        self._betas_buf[0]  = torch.from_numpy(betas.astype(np.float32))
        self._transl_buf[0] = torch.from_numpy(transl.astype(np.float32))
        self._orient_buf[0] = torch.from_numpy(global_orient.astype(np.float32))

        with torch.no_grad():
            output = self.model_single(
                body_pose=self._pose_buf,
                global_orient=self._orient_buf,
                transl=self._transl_buf,
                betas=self._betas_buf,
                return_verts=True
            )

        vertices = output.vertices[0].cpu().numpy()
        joints   = output.joints[0].cpu().numpy()

        return vertices, joints

    # ---------------------------------------------------------
    # Батчевый forward — для оптимизации (вся популяция за раз)
    # ---------------------------------------------------------

    def forward_batch(self, chromosomes):

        n = len(chromosomes)

        # Если размер не совпадает с batch_size — пересоздать модель
        if n != self.batch_size:
            print(f"Warning: batch size mismatch {n} != {self.batch_size}")
            self.batch_size = n
            self.model_batch = create(
                model_path="src",
                model_type="smplx",
                gender="neutral",
                batch_size=n,
                use_pca=False,
                flat_hand_mean=True
            ).to(self.device)

        body_pose     = np.stack([c.body_pose     for c in chromosomes])
        betas         = np.stack([c.betas         for c in chromosomes])
        transl        = np.stack([c.transl        for c in chromosomes])
        global_orient = np.stack([c.global_orient for c in chromosomes])

        pose_t   = torch.from_numpy(body_pose).float().to(self.device)
        betas_t  = torch.from_numpy(betas).float().to(self.device)
        transl_t = torch.from_numpy(transl).float().to(self.device)
        orient_t = torch.from_numpy(global_orient).float().to(self.device)

        with torch.no_grad():
            output = self.model_batch(
                body_pose=pose_t,
                global_orient=orient_t,
                transl=transl_t,
                betas=betas_t,
                return_verts=False
            )

        return output.joints.cpu().numpy()  # (n, 127, 3)