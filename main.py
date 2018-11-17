from image import calculate_image
from render import Renderer

import numpy as np
import tensorflow as tf

from datetime import datetime
import imageio
import sys
import argparse

MAX_GENERATIONS = 200
GENERATION_SIZE = 100
BATCH_SIZE = GENERATION_SIZE  # decrease this if memory is tight, must divide GENERATION_SIZE evenly
INIT_POP_AVERAGE_TRIANGLES = 5
PROB_DEL_TRI = 0.50  # chance any triangle is deleted
PROB_ADD_TRI = 0.50  # change a random triangle is added
GENERATION_CARRYOVER = int(GENERATION_SIZE * 0.1)
np.random.seed()

generation = 0

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


def _calculate_ssim(tfsess, b):
    # requires init_tensors first
    # b is a batch of BATCH_SIZE images
    ssim_v, summary = tfsess.run([_calculate_ssim.tensor, _calculate_ssim.summary], feed_dict={_calculate_ssim.im_b: b})
    tf_log_writer.add_summary(summary, generation)
    return ssim_v


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


def train(tfsess, resolution, target_ssim=0.7, intermediate_path=None):
    global generation
    population = randomly_generate_population(GENERATION_SIZE, INIT_POP_AVERAGE_TRIANGLES, resolution)
    generation = 0
    while True:
        population_fitness = []
        for b in range(GENERATION_SIZE // BATCH_SIZE):
            image_batch = [calculate_image(population[i+b*BATCH_SIZE]) for i in range(BATCH_SIZE)]
            population_fitness.extend(_calculate_ssim(tfsess, image_batch))
            del image_batch

        population_fitness, population = sort_two_lists(population_fitness, population)
        if intermediate_path:
            imageio.imwrite('{}/gen{}.png'.format(intermediate_path, generation), np.uint8(calculate_image(population[0]) * 256))

        population_fitness = np.array(population_fitness)
        print("Gen {}; Best: {}; Mean: {}.".format(generation, population_fitness[0], population_fitness.mean()))

        # check if we have reached our target_loss
        if generation >= MAX_GENERATIONS:
            break  # Not an optimal solution, but we have to stop somewhere
        elif population_fitness.max() >= target_ssim:
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


def init_tensors(tfsess, image, resolution):
    with tf.name_scope('base_image'):
        _calculate_ssim.im_a = tf.constant(np.float32(image).reshape(1, resolution[1], resolution[0], 3) / 256.0)
        im_a = tf.tile(_calculate_ssim.im_a, (BATCH_SIZE, 1, 1, 1))
    with tf.name_scope('input_images'):
        _calculate_ssim.im_b = tf.placeholder(np.float32, shape=(BATCH_SIZE, resolution[1], resolution[0], 3))
        im_b = tf.cast(_calculate_ssim.im_b, np.float32)
    with tf.name_scope('ssim'):
        _calculate_ssim.tensor = tf.image.ssim(im_a, im_b, 1.0)

        with tf.name_scope('summaries'):
            ssim = _calculate_ssim.tensor
            mean = tf.reduce_mean(ssim)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(ssim - mean)))

            max = tf.reduce_max(ssim)
            min = tf.reduce_min(ssim)
            mean_s = tf.summary.scalar('mean', mean)
            stddev_s = tf.summary.scalar('stddev', stddev)
            max_s = tf.summary.scalar('max', max)
            min_s = tf.summary.scalar('min', min)
            ssim_hs = tf.summary.histogram('ssim', ssim)

    _calculate_ssim.summary = tf.summary.merge([mean_s, stddev_s, max_s, min_s, ssim_hs])

    global tf_log_writer
    tf_log_writer = tf.summary.FileWriter('./log/{}'.format(datetime.now()), tfsess.graph)


def main(photo, output, target_ssim=0.7, save_intermediate=False):
    image = imageio.imread(photo)
    resolution = (image.shape[1], image.shape[0]) # because we store in Row-major order

    renderer = Renderer(resolution)

    with tf.Session() as sess:
        init_tensors(sess, image, resolution)
        pop, fitness, generation = train(sess, resolution, target_ssim=target_ssim, intermediate_path=output if save_intermediate else None)
        imageio.imwrite('{}/final.png'.format(output), np.uint8(calculate_image(pop) * 256.0))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Use a genetic algorithm to create a represetnation of an image.")
    # parser.add_argument("source", metavar='src', type=str, help="source image")
    # parser.add_argument("output", metavar='dst', type=str, help="output destination")
    # parser.add_argument()
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] in ['True', 'False'])
