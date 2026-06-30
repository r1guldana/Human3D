# joint_mapper.py
import numpy as np

JOINT_MAP = {
    0:  55,   # Nose
    2:  57,   # Left eye
    5:  56,   # Right eye
    7:  58,   # Left ear
    8:  59,   # Right ear
    11: 16,   # Left shoulder
    12: 17,   # Right shoulder
    13: 18,   # Left elbow
    14: 19,   # Right elbow
    15: 20,   # Left wrist
    16: 21,   # Right wrist
    23:  1,   # Left hip
    24:  2,   # Right hip
    25:  4,   # Left knee
    26:  5,   # Right knee
    27:  7,   # Left ankle
    28:  8,   # Right ankle
    29: 60,   # Left heel
    30: 63,   # Right heel
    31: 10,   # Left foot
    32: 11,   # Right foot
}


def get_corresponding_joints(mp_joints, smpl_joints):

    mp_points   = []
    smpl_points = []

    for mp_idx, smpl_idx in JOINT_MAP.items():
        mp_points.append(mp_joints[mp_idx])
        smpl_points.append(smpl_joints[smpl_idx])

    mp_arr   = np.asarray(mp_points,   dtype=np.float32)
    smpl_arr = np.asarray(smpl_points, dtype=np.float32)

    # Центрируем по бёдрам — убираем сдвиг систем координат
    mp_center   = (mp_joints[23] + mp_joints[24]) / 2
    smpl_center = (smpl_joints[1] + smpl_joints[2]) / 2

    mp_arr   -= mp_center
    smpl_arr -= smpl_center

    return mp_arr, smpl_arr


def joint_error(mp_joints, smpl_joints):

    mp_points, smpl_points = get_corresponding_joints(
        mp_joints, smpl_joints
    )

    return np.mean(
        np.linalg.norm(mp_points - smpl_points, axis=1)
    )


def validate_joint_map(smpl_joints):
    """Вызвать один раз при старте для проверки индексов."""

    print(f"Total SMPL-X joints: {len(smpl_joints)}")

    for mp_idx, smpl_idx in JOINT_MAP.items():
        if smpl_idx >= len(smpl_joints):
            print(f"ERROR: mp_idx={mp_idx} -> smpl_idx={smpl_idx} "
                  f"OUT OF RANGE (max={len(smpl_joints)-1})")
        else:
            print(f"OK   : mp_idx={mp_idx:2d} -> smpl_idx={smpl_idx:3d} "
                  f"pos={smpl_joints[smpl_idx].round(3)}")