#genetic.py
import copy
import random
from dataclasses import dataclass
from src.joint_mapper import get_corresponding_joints

import numpy as np


# --------------------------------------------------------
# Параметры алгоритма
# --------------------------------------------------------

POPULATION_SIZE = 50 #для ускорения, вместо 100
ELITE_COUNT = 5

MUTATION_RATE = 0.3
MUTATION_STD = 0.08

TOURNAMENT_SIZE = 5


# --------------------------------------------------------
# Размеры хромосомы
# --------------------------------------------------------

NUM_BODY_JOINTS = 21
POSE_SIZE = NUM_BODY_JOINTS * 3   # 63
BETAS_SIZE = 10


# --------------------------------------------------------
# Хромосома
# --------------------------------------------------------

@dataclass
class Chromosome:

    global_orient: np.ndarray      # (3,)
    body_pose: np.ndarray          # (69,)
    betas: np.ndarray              # (10,)
    transl: np.ndarray             # (3,)
    fitness: float = np.inf

    @staticmethod
    def random():

        return Chromosome(

            transl=np.random.uniform(
                -1.0,
                1.0,
                3
            ),

            global_orient=np.random.uniform(
                -np.pi,
                np.pi,
                3
            ),

            body_pose=np.random.uniform(
                -0.5,
                0.5,
                POSE_SIZE
            ),

            betas=np.random.uniform(
                -2.0,
                2.0,
                BETAS_SIZE
            ),

            fitness=np.inf
        )

    def clone(self):

        return copy.deepcopy(self)


# --------------------------------------------------------
# Популяция
# --------------------------------------------------------

class Population:

    def __init__(self, size=POPULATION_SIZE):

        self.individuals = [

            Chromosome.random()

            for _ in range(size)

        ]

    def sort(self):

        self.individuals.sort(
            key=lambda x: x.fitness
        )

    def best(self):

        self.sort()

        return self.individuals[0]

    def elites(self):

        self.sort()

        return [

            x.clone()

            for x in self.individuals[:ELITE_COUNT]

        ]


# --------------------------------------------------------
# Генетический алгоритм
# --------------------------------------------------------

class GeneticAlgorithm:

    def __init__(self):

        self.population = Population()

    # ----------------------------------------------------

    def evaluate(
        self,
        fitness_function,
        measurements
    ):

        for individual in self.population.individuals:

            individual.fitness = fitness_function(
                individual,
                measurements
            )
    
    def evaluate_batch(self, smpl_model, frame):
        """
        Оценивает всю популяцию за один batched forward pass.
        """

        individuals = self.population.individuals
        n = len(individuals)

        # Батчевый forward
        joints_batch = smpl_model.forward_batch(individuals)  # (n, 127, 3)

        mp_joints = frame["joints_3d"]  # (33, 3)

        for i, individual in enumerate(individuals):

            mp_pts, smpl_pts = get_corresponding_joints(
                mp_joints,
                joints_batch[i]   # (127, 3)
            )

            error = np.mean(
                np.linalg.norm(mp_pts - smpl_pts, axis=1)
            )

            error += 0.01 * np.mean(individual.betas     ** 2)
            error += 0.01 * np.mean(individual.body_pose ** 2)

            individual.fitness = float(error)

    # ----------------------------------------------------

    def tournament(self):

        candidates = random.sample(

            self.population.individuals,

            TOURNAMENT_SIZE

        )

        candidates.sort(

            key=lambda x: x.fitness

        )

        return candidates[0].clone()

    # ----------------------------------------------------

    def crossover(

        self,

        parent1,

        parent2

    ):

        child = parent1.clone()

        mask = np.random.rand(3) < 0.5
        child.transl[mask] = parent2.transl[mask]

        mask = np.random.rand(3) < 0.5
        child.global_orient[mask] = parent2.global_orient[mask]

        mask = np.random.rand(POSE_SIZE) < 0.5
        child.body_pose[mask] = parent2.body_pose[mask]

        mask = np.random.rand(BETAS_SIZE) < 0.5
        child.betas[mask] = parent2.betas[mask]

        child.fitness = np.inf

        return child

    # ----------------------------------------------------

    def mutate(self, individual):

        # Каждый элемент мутирует независимо
        mask = np.random.rand(3) < MUTATION_RATE
        individual.transl[mask] += np.random.normal(
            0, MUTATION_STD, mask.sum()
        )

        mask = np.random.rand(3) < MUTATION_RATE
        individual.global_orient[mask] += np.random.normal(
            0, MUTATION_STD, mask.sum()
        )

        mask = np.random.rand(POSE_SIZE) < MUTATION_RATE
        individual.body_pose[mask] += np.random.normal(
            0, MUTATION_STD, mask.sum()
        )

        mask = np.random.rand(BETAS_SIZE) < MUTATION_RATE
        individual.betas[mask] += np.random.normal(
            0, MUTATION_STD, mask.sum()
        )

        return individual
    
        # ----------------------------------------------------
    # Создание нового поколения
    # ----------------------------------------------------

    def evolve(self):

        new_population = []

        # Элитизм
        new_population.extend(
            self.population.elites()
        )

        while len(new_population) < POPULATION_SIZE:

            parent1 = self.tournament()
            parent2 = self.tournament()

            child = self.crossover(
                parent1,
                parent2
            )

            child = self.mutate(child)

            new_population.append(child)

        self.population.individuals = new_population

    # ----------------------------------------------------

    def run(
        self,
        fitness_function,
        measurements,
        generations=20,
        patience=5        # стоп если N поколений без улучшения
    ):

        self.evaluate(fitness_function, measurements)

        best         = self.population.best()
        best_fitness = best.fitness
        no_improve   = 0

        for generation in range(1, generations + 1):

            self.evolve()
            self.evaluate(fitness_function, measurements)

            best = self.population.best()

            if best.fitness < best_fitness - 1e-5:
                best_fitness = best.fitness
                no_improve   = 0
            else:
                no_improve  += 1

            print(f"  Gen {generation:3d}  "
                f"fitness={best.fitness:.4f}  "
                f"patience={no_improve}/{patience}")

            if no_improve >= patience:
                print(f"  Early stop at gen {generation}")
                break

        return self.population.best()
    
    def run_batch(self, smpl_model, frame, generations=20, patience=5):
        """
        Версия run() которая использует evaluate_batch вместо evaluate.
        """

        # Первая оценка
        self.evaluate_batch(smpl_model, frame)

        best         = self.population.best()
        best_fitness = best.fitness
        no_improve   = 0

        print(f"  Gen   0  fitness={best_fitness:.4f}")

        for generation in range(1, generations + 1):

            self.evolve()
            self.evaluate_batch(smpl_model, frame)  # батч вместо поодиночке

            best = self.population.best()

            if best.fitness < best_fitness - 1e-5:
                best_fitness = best.fitness
                no_improve   = 0
            else:
                no_improve  += 1

            print(f"  Gen {generation:3d}  "
                f"fitness={best.fitness:.4f}  "
                f"patience={no_improve}/{patience}")

            if no_improve >= patience:
                print(f"  Early stop at gen {generation}")
                break

        return self.population.best()
    
    # --------------------------------------------------------
# Временная fitness
# --------------------------------------------------------

def fitness(individual, measurements):

    target = np.zeros(POSE_SIZE)

    error = np.mean(
        (individual.body_pose - target) ** 2
    )

    error += np.mean(
        individual.transl ** 2
    )

    error += 0.1 * np.mean(
        individual.betas ** 2
    )

    return error

# --------------------------------------------------------
# Точка входа
# --------------------------------------------------------

if __name__ == "__main__":

    ga = GeneticAlgorithm()

    best = ga.run(
        fitness,
        None,
        generations=100
    )

    print()

    print("Optimization finished")

    print("Fitness:", best.fitness)

    print("Translation")

    print(best.transl)

    print()

    print("Rotation")

    print(best.global_orient)