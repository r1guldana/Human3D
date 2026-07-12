import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

BASE_DIR   = Path("d:/Disseratation/Human3D")
VIDEO_PATH = BASE_DIR / "data/videos/vidR.mp4"
MODEL_PATH = BASE_DIR / "src/pose_landmarker_full.task"


class PoseDetector:

    def __init__(self, model_path=MODEL_PATH):

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    # -------------------------------------------------

    def process_video(
        self,
        video_path,
        frame_start=0,    # <- первый нужный кадр
        frame_end=None    # <- последний нужный кадр (None = до конца)
    ):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS)
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Нормализуем frame_end
        if frame_end is None:
            frame_end = frame_count
        frame_end = min(frame_end, frame_count)

        print(f"Total frames : {frame_count}")
        print(f"Processing   : {frame_start}..{frame_end} "
              f"({frame_end - frame_start} frames)")
        print(f"FPS          : {fps}")
        print(f"Resolution   : {width}x{height}")

        # Перемотать до нужного кадра через CAP_PROP_POS_FRAMES
        # (быстрее чем читать и выбрасывать)
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        video    = []
        frame_id = frame_start

        while frame_id < frame_end:

            ret, frame = cap.read()

            if not ret:
                print(f"Warning: stopped at frame {frame_id}")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # timestamp должен быть монотонно возрастающим
            timestamp = int(frame_id * 1000 / fps)

            result = self.detector.detect_for_video(mp_image, timestamp)

            joints2d   = np.zeros((33, 2), dtype=np.float32)
            joints3d   = np.zeros((33, 3), dtype=np.float32)
            visibility = np.zeros(33,      dtype=np.float32)
            segmentation_mask = None

            if len(result.pose_landmarks) > 0:

                pose2d = result.pose_landmarks[0]
                pose3d = result.pose_world_landmarks[0]

                for i in range(33):
                    joints2d[i] = (
                        pose2d[i].x * width,
                        pose2d[i].y * height
                    )
                    joints3d[i] = (
                        pose3d[i].x,
                        -pose3d[i].y,  # инвертирование координат Y
                        pose3d[i].z
                    )
                    visibility[i] = (
                        pose2d[i].visibility
                        if hasattr(pose2d[i], "visibility")
                        else 1.0
                    )

            if (
                hasattr(result, "segmentation_masks")
                and result.segmentation_masks
            ):
                segmentation_mask = (
                    result.segmentation_masks[0]
                    .numpy_view()
                    .copy()
                )

            video.append({
                "frame_id":         frame_id,
                "time":             frame_id / fps,
                "frame":            frame.copy(),
                "width":            width,
                "height":           height,
                "joints_2d":        joints2d,
                "joints_3d":        joints3d,
                "visibility":       visibility,
                "segmentation_mask": segmentation_mask,
            })

            frame_id += 1

        cap.release()

        print(f"Loaded {len(video)} frames into memory")

        return video

    # -------------------------------------------------

    def close(self):
        self.detector.close()


# -------------------------------------------------

if __name__ == "__main__":

    detector = PoseDetector(MODEL_PATH)

    frames = detector.process_video(
        VIDEO_PATH,
        frame_start=100,
        frame_end=200
    )

    detector.close()

    print()
    print("Frames processed:", len(frames))
    print("First frame keys:", list(frames[0].keys()))
    print("2D shape:", frames[0]["joints_2d"].shape)
    print("3D shape:", frames[0]["joints_3d"].shape)