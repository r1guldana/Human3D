import json

import numpy as np

from src.genetic import GeneticAlgorithm
from src.joint_mapper import get_corresponding_joints


class Optimizer:
    def __init__(self, smpl_model):
        self.smpl = smpl_model

    def fitness(self, chromosome, frame):
        vertices, joints = self.smpl.forward(
            body_pose=chromosome.body_pose,
            betas=chromosome.betas,
            transl=chromosome.transl,
            global_orient=chromosome.global_orient,
        )

        mp_points, smpl_points = get_corresponding_joints(
            frame["joints_3d"],
            joints,
        )

        joint_error = np.mean(np.linalg.norm(mp_points - smpl_points, axis=1))
        beta_reg = 0.005 * np.mean(chromosome.betas**2)
        pose_reg = 0.005 * np.mean(chromosome.body_pose**2)
        transl_reg = 0.001 * np.mean(chromosome.transl**2)

        return joint_error + beta_reg + pose_reg + transl_reg

    def optimize_frame(self, frame, generations=20, warm_start=None):
        ga = GeneticAlgorithm(total_generations=generations)

        if warm_start is not None:
            half = len(ga.population.individuals) // 2
            for i in range(half):
                mutated = warm_start.clone()
                ga.mutate(mutated)
                ga.population.individuals[i] = mutated

        ga.evaluate_batch(self.smpl, frame)

        best = ga.population.best()
        print(f"  Gen   0  fitness={best.fitness:.4f}")

        no_improve_count = 0
        best_fitness = best.fitness
        PATIENCE = 10

        for gen in range(1, generations + 1):
            ga.evolve()
            ga.evaluate_batch(self.smpl, frame)

            best = ga.population.best()
            print(f"  Gen {gen:3d}  fitness={best.fitness:.4f}")
            if best.fitness < best_fitness - 1e-5:
                best_fitness = best.fitness
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= PATIENCE:
                print(f"  Early stop at gen {gen}")
                break
        return ga.population.best()

    def optimize_video(self, frames, generations=80, output_dir=None):
        result = []
        prev_best = None

        for i, frame in enumerate(frames):
            gens = generations if i == 0 else 20

            print(
                f"\nFrame {i + 1}/{len(frames)} "
                f"| generations={gens} "
                f"| warm_start={prev_best is not None}"
            )

            best = self.optimize_frame(frame, generations=gens, warm_start=prev_best)

            prev_best = best
            result.append(best)

            print(f"Frame {i + 1} fitness={best.fitness:.6f}")
            if output_dir is not None:
                checkpoint = {
                    "frame":        i,
                    "fitness":      float(best.fitness),
                    "body_pose":    best.body_pose.tolist(),
                    "betas":        best.betas.tolist(),
                    "transl":       best.transl.tolist(),
                    "global_orient":best.global_orient.tolist()
                }
                with open(f"{output_dir}/frame_{i:04d}.json", "w") as f:
                    json.dump(checkpoint, f)

        return result