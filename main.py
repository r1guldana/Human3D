from src.pose_detector import PoseDetector
from src.skeleton import Skeleton
from src.optimizer import Optimizer
from pathlib import Path
import numpy as np
#from src.joint_mapper import validate_joint_map

import time

BASE_DIR    = Path("C:/Users/user/Desktop/Mariya/projects/compv/Human3D")
VIDEO_PATH  = BASE_DIR / "data/videos/vidR.mp4"
OUTPUT_CSV  = BASE_DIR / "data/poses/poses.csv"
MODEL_PATH  = BASE_DIR / "src/pose_landmarker_full.task"
OUTPUT_DIR = BASE_DIR / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 100
FRAME_END   = 200

def main():
    print("Loading Pose Detector...")
    detector = PoseDetector(model_path=MODEL_PATH)

    print("Loading SMPL-X...")
    '''
    smpl_single = Skeleton(batch_size=1)
    _, test_joints = smpl_single.forward(
        np.zeros(63), np.zeros(10), np.zeros(3), np.zeros(3)
    )
    validate_joint_map(test_joints)  # убедиться что все индексы верны
    '''
    smpl = Skeleton(batch_size=60)

    print("Loading Optimizer...")
    optimizer = Optimizer(smpl)

    print("Reading video...")
    detector = PoseDetector(model_path=MODEL_PATH)
    try:
        frames = detector.process_video(
            str(VIDEO_PATH),
            frame_start=FRAME_START,
            frame_end=FRAME_END
        )
    finally:
        detector.close()

    print(f"Frames loaded: {len(frames)}")

    print("Starting optimization...")

    result = optimizer.optimize_video(
        frames,
        generations=80,
        output_dir=OUTPUT_DIR
    )

    print()
    print("Optimization finished.")

    if not result:
        raise RuntimeError("Optimizer returned empty result")

    best = result[0]

    smpl_single = Skeleton(batch_size=1)

    vertices, joints = smpl_single.forward(
        body_pose=best.body_pose,
        betas=best.betas,
        transl=best.transl,
        global_orient=best.global_orient
    )
    print()
    print(vertices.shape)
    print(joints.shape)

    np.save(OUTPUT_DIR / "vertices.npy", vertices)
    np.save(OUTPUT_DIR / "joints.npy", joints)

    print("Saved:")
    print("vertices.npy")
    print("joints.npy")


if __name__ == "__main__":
    main()