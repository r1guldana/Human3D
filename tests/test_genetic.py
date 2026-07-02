import numpy as np

from src.genetic import GeneticAlgorithm, Chromosome, POPULATION_SIZE


def test_mutation_keeps_shapes():
    ga = GeneticAlgorithm(total_generations=5)
    ind = Chromosome.random()
    mutated = ga.mutate(ind)

    assert mutated.body_pose.shape == ind.body_pose.shape
    assert mutated.betas.shape == ind.betas.shape
    assert mutated.transl.shape == ind.transl.shape
    assert mutated.global_orient.shape == ind.global_orient.shape


def test_population_size_matches_config():
    ga = GeneticAlgorithm(total_generations=5)
    assert len(ga.population.individuals) == POPULATION_SIZE


def test_custom_population_and_mutation_settings():
    ga = GeneticAlgorithm(
        total_generations=5,
        population_size=12,
        mutation_rate_init=0.2,
        mutation_rate_min=0.02,
    )
    assert len(ga.population.individuals) == 12
    assert ga.mutation_rate_init == 0.2
    assert ga.mutation_rate_min == 0.02
