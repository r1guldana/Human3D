import numpy as np
from src.genetic import GeneticAlgorithm, Chromosome, POPULATION_SIZE
from src.joint_mapper import get_corresponding_joints


class Optimizer:

    def __init__(self, smpl_model):
        self.smpl = smpl_model

    # -------------------------------------------------

    def fitness(self, chromosome, frame):

        vertices, joints = self.smpl.forward(
            body_pose=chromosome.body_pose,
            betas=chromosome.betas,
            transl=chromosome.transl,
            global_orient=chromosome.global_orient
        )

        mp_points, smpl_points = get_corresponding_joints(
            frame["joints_3d"],
            joints
        )

        error = np.mean(
            np.linalg.norm(mp_points - smpl_points, axis=1)
        )

        error += 0.01 * np.mean(chromosome.betas ** 2)
        error += 0.01 * np.mean(chromosome.body_pose ** 2)

        return error

    # -------------------------------------------------

    def optimize_frame(
        self,
        frame,
        generations=20,
        warm_start=None   # <- лучшая хромосома предыдущего кадра
    ):
        ga = GeneticAlgorithm()  # свежий ГА на каждый кадр

        # Тёплый старт: заполняем популяцию мутациями предыдущего решения
        if warm_start is not None:
            half = len(ga.population.individuals) // 2
            for i in range(half):
                mutated = warm_start.clone()
                ga.mutate(mutated)
                ga.population.individuals[i] = mutated

        # Первая оценка
        ga.evaluate_batch(self.smpl, frame)

        best = ga.population.best()
        print(f"  Gen   0  fitness={best.fitness:.4f}")

        for gen in range(1, generations + 1):

            ga.evolve()
            ga.evaluate_batch(self.smpl, frame)  # <- батч вместо по одному

            best = ga.population.best()
            print(f"  Gen {gen:3d}  fitness={best.fitness:.4f}")

        return ga.population.best()

    # -------------------------------------------------

    def optimize_video(self, frames, generations=80):

        result = []
        prev_best = None  # храним лучшее решение предыдущего кадра

        for i, frame in enumerate(frames):

            # Первый кадр — полная оптимизация
            # Остальные — тёплый старт + меньше поколений
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