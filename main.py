from src.pose_detector import PoseDetector
from src.skeleton import Skeleton
from src.optimizer import Optimizer
from pathlib import Path
import numpy as np

BASE_DIR    = Path("d:/Disseratation/Human3D")
VIDEO_PATH  = BASE_DIR / "data/videos/vidR.mp4"
MODEL_PATH  = BASE_DIR / "src/pose_landmarker_full.task"
OUTPUT_DIR  = BASE_DIR / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 100
FRAME_END   = 200


def main():

    # ── Детекция поз ──────────────────────────────
    print("Loading Pose Detector...")
    detector = PoseDetector(model_path=MODEL_PATH)

    try:
        print("Reading video...")
        frames = detector.process_video(
            str(VIDEO_PATH),
            frame_start=FRAME_START,
            frame_end=FRAME_END
        )
    finally:
        detector.close()

    print(f"Frames loaded: {len(frames)}")

    # ── SMPL-X — один экземпляр batch_size=50 ─────
    print("Loading SMPL-X...")
    smpl = Skeleton(batch_size=50)

    # ── Оптимизация ───────────────────────────────
    print("Loading Optimizer...")
    optimizer = Optimizer(smpl)

    print("Starting optimization...")
    result = optimizer.optimize_video(frames, generations=80)

    if not result:
        raise RuntimeError("Optimizer returned empty result")

    print("\nOptimization finished.")

    # ── Финальный результат — используем ТОТ ЖЕ smpl ─
    # forward() работает с batch_size=50 через model_single
    best = result[0]

    vertices, joints = smpl.forward(
        body_pose=best.body_pose,
        betas=best.betas,
        transl=best.transl,
        global_orient=best.global_orient
    )

    print(f"Vertices : {vertices.shape}")
    print(f"Joints   : {joints.shape}")

    np.save(OUTPUT_DIR / "vertices.npy", vertices)
    np.save(OUTPUT_DIR / "joints.npy",   joints)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()