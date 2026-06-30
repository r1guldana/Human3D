# Human3D - Video to SMPL-X Pose Reconstruction

Human3D reconstructs 3D human body pose and shape from monocular RGB video. It detects 2D/3D body landmarks with MediaPipe Pose, then fits the SMPL-X parametric body model to those landmarks frame-by-frame using a custom genetic algorithm (GA) optimizer.

## Overview

The pipeline works in three stages:

1. **Pose Detection**: Extracts 3D body landmarks from each video frame using MediaPipe's Pose Landmarker.
2. **Model Fitting**: Optimizes SMPL-X parameters (global orientation, body pose, shape betas, translation) per frame so that the model's joints align with the detected landmarks. Optimization is performed with a genetic algorithm rather than gradient descent, using batched forward passes through SMPL-X for speed.
3. **Temporal Warm-Start**: After the first frame is fully optimized, subsequent frames seed their population with mutated copies of the previous frame's best solution, reducing the number of generations needed and improving temporal consistency.
## How It Works

### Pose Detection (`pose_detector.py`)
Uses MediaPipe's Pose Landmarker to extract 3D world-space joint coordinates (33 landmarks) for each frame in a given frame range of the input video.

### SMPL-X Model (`skeleton.py`)
Wraps the SMPL-X body model, producing mesh vertices and 3D joints from a set of pose parameters:
- `global_orient` (3,): root rotation
- `body_pose` (63,): 21 body joint rotations
- `betas` (10,): shape coefficients
- `transl` (3,): root translation

Supports both single-instance and batched forward passes for efficient population evaluation.

### Genetic Optimizer (`genetic.py`)
Each candidate solution ("chromosome") encodes the SMPL-X parameters above. The GA evolves a population of 50 individuals using:
- **Elitism**: top 5 individuals carried over unchanged each generation
- **Tournament selection** (size 5)
- **Uniform crossover** per parameter group
- **Gaussian mutation** (rate 0.15, std 0.05)

Fitness is computed as the mean Euclidean distance between corresponding MediaPipe and SMPL-X joints (via `joint_mapper.py`), plus small regularization terms on betas and body pose to discourage unrealistic shapes/poses.

### Video Optimization (`optimizer.py`)
- Frame 1: full optimization (e.g. 80 generations) from a random population.
- Subsequent frames: warm-started from the previous frame's best solution (mutated copies fill half the population) and optimized for fewer generations (e.g. 20), since the result should be close to the previous frame.

### Joint Mapping (`joint_mapper.py`)
Defines the correspondence between MediaPipe's 33 pose landmarks and SMPL-X's joint set, so that the two skeletons can be compared during fitness evaluation.

## Output

For the last optimized frame, the pipeline saves:
- `vertices.npy`: SMPL-X mesh vertices
- `joints.npy`: SMPL-X joint positions

These can be visualized or used for downstream analysis (motion analysis, animation retargeting, etc.).

## Requirements

- Python 3.x
- numpy
- opencv-python
- mediapipe
- torch
- smplx


## Usage

1. Place your input video at `data/videos/vidR.mp4` (or update `VIDEO_PATH` in `main.py`).
2. Adjust `FRAME_START` / `FRAME_END` in `main.py` to select the frame range to process.


Estimated SMPL-X parameters and resulting vertices/joints will be saved to `data/output/`.

## Notes

- The genetic algorithm approach avoids the need for differentiable rendering or analytic gradients, at the cost of being slower per-frame than gradient-based fitters (e.g. SMPLify-X).
- Batched fitness evaluation (`evaluate_batch` / `forward_batch`) is used to significantly speed up the GA by evaluating the entire population in a single forward pass through SMPL-X.
- This project is part of an ongoing dissertation/research effort and is a work in progress.
