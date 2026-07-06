import numpy as np
from src.genetic import GeneticAlgorithm, Chromosome, POPULATION_SIZE, POSE_SIZE
from src.joint_mapper import get_corresponding_joints


class Optimizer:

    def __init__(self, smpl_model):
        self.smpl = smpl_model

    # -------------------------------------------------
    # Одиночная проверка — только для отладки
    # -------------------------------------------------

    def fitness(self, chromosome, frame):

        _, joints = self.smpl.forward(
            body_pose=chromosome.body_pose,
            betas=chromosome.betas,
            transl=chromosome.transl,
            global_orient=chromosome.global_orient
        )

        mp_points, smpl_points = get_corresponding_joints(
            frame["joints_3d"], joints
        )

        error = np.mean(
            np.linalg.norm(mp_points - smpl_points, axis=1)
        )

        error += 0.01 * np.mean(chromosome.betas    ** 2)
        error += 0.01 * np.mean(chromosome.body_pose ** 2)

        return error

    # -------------------------------------------------
    # Warm start с разным уровнем мутации
    # -------------------------------------------------

    def _apply_warm_start(self, ga, warm_start):
        """
        Заполняем всю популяцию мутациями warm_start.
        Разный масштаб мутации — сохраняем разнообразие.
        """

        individuals = ga.population.individuals
        n           = len(individuals)

        for i, individual in enumerate(individuals):

            mutated = warm_start.clone()

            # Масштаб мутации растёт от 0.02 до 0.25
            # Первые особи — близко к warm_start (эксплуатация)
            # Последние — далеко (исследование)
            scale = 0.02 + (i / n) * 0.23

            mutated.body_pose     += np.random.normal(0, scale, POSE_SIZE)
            mutated.global_orient += np.random.normal(0, scale, 3)
            mutated.transl        += np.random.normal(0, scale, 3)
            mutated.betas         += np.random.normal(0, scale * 0.3, 10)
            mutated.fitness        = np.inf

            individuals[i] = mutated

    # -------------------------------------------------

    def optimize_frame(
        self,
        frame,
        generations=20,
        warm_start=None
    ):
        # Рестарты убраны — warm_start достаточно

        ga = GeneticAlgorithm()

        if warm_start is not None:
            self._apply_warm_start(ga, warm_start)

        best = ga.run_batch(
            smpl_model=self.smpl,
            frame=frame,
            generations=generations,
            patience=8    # было 5 — дать больше времени
        )

        return best

    # -------------------------------------------------

    def optimize_video(self, frames, generations=80):

        result    = []
        prev_best = None

        for i, frame in enumerate(frames):

            gens = generations if i == 0 else 20

            print(f"\nFrame {i+1}/{len(frames)} "
                  f"| generations={gens} "
                  f"| warm_start={prev_best is not None}")

            best = self.optimize_frame(
                frame,
                generations=gens,
                warm_start=prev_best
            )

            prev_best = best
            result.append(best)

            print(f"Frame {i+1} fitness={best.fitness:.6f}")

        return result