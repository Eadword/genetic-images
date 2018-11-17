from image import calculate_image
from render import Renderer

import numpy as np
import tensorflow as tf

import imageio
import sys
import argparse

MAX_GENERATIONS = 5
GENERATION_SIZE = 80
INIT_POP_AVERAGE_TRIANGLES = 5
PROB_DEL_TRI = 0.50  # chance any triangle is deleted
PROB_ADD_TRI = 0.50  # change a random triangle is added
GENERATION_CARRYOVER = int(GENERATION_SIZE * 0.25)
np.random.seed(908759798)

# A Pop is a tuple of (Triangles, Colors) where a color is an RGBA value stored as 4 uint8s
# A Triangle is a (n, 2) shaped array of coordinates


def randomly_generate_triangle(resolution):
    return np.int32(np.random.rand(3, 2) * resolution), np.uint8(np.random.rand(4) * 256)


def randomly_generate_pop(triangles, resolution):
    tl = []
    cl = []
    for _ in range(triangles):
        t, c = randomly_generate_triangle(resolution)
        tl.append(t)
        cl.append(c)
    return tl, cl


def randomly_generate_population(pop_size, avg_triangles, resolution):
    return [
        randomly_generate_pop(
            np.random.randint(min(1, avg_triangles // 2), avg_triangles * 2),
            resolution
        )
        for _ in range(pop_size)
    ]


def calculate_loss(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.sum(np.sqrt((a - b)**2))


def copy_pop(pop):
    return (pop[0].copy(), pop[1].copy())


def mutate_pop(pop, resolution):
    # Two types of mutation 1) remove a triangle, 2) add a triangle
    while len(pop[0]) > 1 and np.random.rand() < PROB_DEL_TRI:
        choice = np.random.randint(0, len(pop[0]))
        del pop[0][choice]
        del pop[1][choice]

    while np.random.rand() < PROB_ADD_TRI:
        verts, color = randomly_generate_triangle(resolution)
        pop[0].append(verts)
        pop[1].append(color)


def sort_two_lists(a, b):
    assert len(a) == len(b)
    indexes = list(range(len(a)))
    indexes.sort(key=a.__getitem__, reverse=True)
    return list(map(a.__getitem__, indexes)), list(map(b.__getitem__, indexes))


def train(tfsess, image, target_loss=1.0, intermediate_path=None):
    resolution = image.shape[:2]
    population = randomly_generate_population(GENERATION_SIZE, INIT_POP_AVERAGE_TRIANGLES, resolution)
    generation = 0
    while True:
        population_fitness = [
            1 / calculate_loss(image, calculate_image(pop))
            for pop in population
        ]

        population_fitness, population = sort_two_lists(population_fitness, population)
        population = list(population)
        population_fitness = np.array(population_fitness)
        print("Gen {}; Best: {}; Mean: {}; Median: {}; SDEV: {}.".format(generation, population_fitness[0], population_fitness.mean(), population_fitness[GENERATION_SIZE//2], population_fitness.std()))

        # check if we have reached our target_loss
        if generation >= MAX_GENERATIONS:
            break  # Not an optimal solution, but we have to stop somewhere
        elif 1 / population_fitness.max() <= target_loss:
            break

        # drop the lowest pops
        population = population[:GENERATION_CARRYOVER]

        # duplicate the top members to fill the gap
        n_babies = GENERATION_SIZE - GENERATION_CARRYOVER
        for i in range(n_babies):
            population.append(copy_pop(population[i]))

        # randomly mutate all individuals
        for i in range(GENERATION_SIZE):
            mutate_pop(population[i], resolution)

        generation+=1

    return population[0], population_fitness[0], generation


def main(photo, output, target_loss=1.0, save_intermediate=False):
    image = imageio.imread(photo)
    resolution = image.shape[:2]

    renderer = Renderer(resolution, hidden=False)

    with tf.Session() as sess:
        pop, fitness, generation = train(sess, image, target_loss=target_loss, intermediate_path=output if save_intermediate else None)
        imageio.imwrite('{}/final.png'.format(output), calculate_image(pop))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Use a genetic algorithm to create a represetnation of an image.")
    # parser.add_argument("source", metavar='src', type=str, help="source image")
    # parser.add_argument("output", metavar='dst', type=str, help="output destination")
    # parser.add_argument()
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] in ['True', 'False'])
