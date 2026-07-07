# fitness.py
import numpy as np
from src.joint_mapper import get_corresponding_joints

# --------------------------------------------------------
# Веса компонентов fitness
# --------------------------------------------------------

WEIGHTS = {
    "joint_distance":   1.0,    # основная метрика — расстояние между суставами
    "pose_prior":       0.01,   # штраф за неестественную позу
    "shape_prior":      0.01,   # штраф за экстремальную форму тела
    "symmetry":         0.05,    # штраф за асимметрию (пока выключен)
    "velocity":         0.1,    # штраф за резкие движения (пока выключен)
}


# --------------------------------------------------------
# Компоненты fitness — каждый можно дорабатывать отдельно
# --------------------------------------------------------

def joint_distance(chromosome, mp_joints, smpl_joints):
    """
    Основная метрика — среднее расстояние между
    суставами MediaPipe и SMPL-X в метрах.
    """

    mp_pts, smpl_pts = get_corresponding_joints(
        mp_joints,
        smpl_joints
    )

    distances = np.linalg.norm(mp_pts - smpl_pts, axis=1)

    return np.mean(distances)


def pose_prior(chromosome):
    """
    Штраф за неестественную позу.
    Большие углы суставов = неестественно.
    """

    return np.mean(chromosome.body_pose ** 2)


def shape_prior(chromosome):
    """
    Штраф за экстремальные параметры формы тела.
    betas далеко от 0 = нестандартное тело.
    """

    return np.mean(chromosome.betas ** 2)


def symmetry_prior(chromosome):
    """
    Штраф за асимметрию левой и правой стороны.
    Пока не используется — WEIGHTS['symmetry'] = 0.0

    SMPL-X body_pose: 21 сустав × 3 угла
    Левые суставы:  1, 4, 7, 10, 13, 16, 18, 20
    Правые суставы: 2, 5, 8, 11, 14, 17, 19, 21
    """

    pose = chromosome.body_pose.reshape(21, 3)

    left_idx  = [0, 3, 6,  9, 12, 15, 17, 19]
    right_idx = [1, 4, 7, 10, 13, 16, 18, 20]

    left  = pose[left_idx]
    right = pose[right_idx]

    # Зеркальная симметрия по оси X
    right_mirrored = right * np.array([-1, 1, 1])

    return np.mean((left - right_mirrored) ** 2)


def velocity_prior(chromosome, prev_chromosome):
    """
    Штраф за резкое изменение позы между кадрами.
    Пока не используется — WEIGHTS['velocity'] = 0.0

    Нужно передавать предыдущую хромосому.
    """

    if prev_chromosome is None:
        return 0.0

    pose_diff   = chromosome.body_pose   - prev_chromosome.body_pose
    transl_diff = chromosome.transl      - prev_chromosome.transl
    orient_diff = chromosome.global_orient - prev_chromosome.global_orient

    return (
        np.mean(pose_diff   ** 2) +
        np.mean(transl_diff ** 2) +
        np.mean(orient_diff ** 2)
    )


# --------------------------------------------------------
# Главная функция — собирает все компоненты
# --------------------------------------------------------

def compute_fitness(
    chromosome,
    mp_joints,
    smpl_joints,
    prev_chromosome=None   # для velocity_prior
):
    """
    Итоговый fitness = взвешенная сумма всех компонентов.

    Чтобы добавить новый компонент:
    1. Написать функцию выше
    2. Добавить вес в WEIGHTS
    3. Добавить строку в compute_fitness
    """

    components = {}

    components["joint_distance"] = joint_distance(
        chromosome, mp_joints, smpl_joints
    )

    components["pose_prior"] = pose_prior(chromosome)

    components["shape_prior"] = shape_prior(chromosome)

    components["symmetry"] = symmetry_prior(chromosome)

    components["velocity"] = velocity_prior(
        chromosome, prev_chromosome
    )

    # Взвешенная сумма
    total = sum(
        WEIGHTS[key] * value
        for key, value in components.items()
    )

    return total, components


# --------------------------------------------------------
# Батчевая версия — для evaluate_batch в genetic.py
# --------------------------------------------------------

def compute_fitness_batch(
    chromosomes,
    mp_joints,
    joints_batch,          # (n, 127, 3) — уже посчитанные SMPL joints
    prev_chromosome=None
):
    """
    Считает fitness для всей популяции сразу.
    joints_batch уже получен через forward_batch — не считаем повторно.
    """

    fitnesses   = []
    n           = len(chromosomes)

    for i, chromosome in enumerate(chromosomes):

        total, _ = compute_fitness(
            chromosome=chromosome,
            mp_joints=mp_joints,
            smpl_joints=joints_batch[i],
            prev_chromosome=prev_chromosome
        )

        fitnesses.append(total)

    return fitnesses  # list of float, длина n