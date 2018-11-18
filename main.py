from image import calculate_image
from render import Renderer

import numpy as np
import numpy.random as random
import tensorflow as tf

from datetime import datetime
import imageio
import sys
from copy import deepcopy, copy
import argparse

MAX_GENERATIONS = 200
GENERATION_SIZE = 100
BATCH_SIZE = GENERATION_SIZE  # decrease this if memory is tight, must divide GENERATION_SIZE evenly
INIT_AVERAGE_TRIANGLES = 5
GENERATION_CARRYOVER = int(GENERATION_SIZE * 0.1)

DIST_MATCHING_DIFF_COST = 0.4
DIST_EXCESS_COST = 1.0
DIST_DISJOINT_COST = 1.0
MAX_DISTANCE_FOR_BREEDING = 0.8
INTERSPECIES_BREEDING_RATE = 0.001

PROB_DEL_TRI = 0.50  # chance any triangle is deleted
PROB_ADD_TRI = 0.50  # change a random triangle is added

PROB_MUT_TRI = 0.05  # chance each triangle is mutated
MUT_TRI_VERT_MEAN = 0.02  # on average, move x*resolution
MUT_TRI_COLOR_MEAN = 1.0  # on average, change each color channel by x

PROB_ALTER_ORDERING = 0.001  # chance any two triangles are swapped thus changing render order

random.seed()

generation = 0
resolution = (0, 0)


class Individual:
    """
    Designed to store triangles as (3, 2) shaped int32 arrays, colors as (4,) shaped uint8 arrays, and trait_ids as ints.
    To get the cross of of two individuals: cross = a @ b (== b @ a)
    To get the compatibility of two individuals: comp = a % b (== b % a)
    """
    def __init__(self, triangles=None, colors=None, trait_ids=None):
        if triangles:
            assert len(triangles) == len(colors) == len(trait_ids)
            self.triangles = triangles
            self.colors = colors
            self.trait_ids = trait_ids
        else:
            self.triangles = []
            self.colors = []
            self.trait_ids = []
        self.fitness = 0.0

    @staticmethod
    def generate(n_triangles):
        tl = []
        cl = []
        for _ in range(n_triangles):
            t, c = randomly_generate_triangle()
            tl.append(t)
            cl.append(c)
        ids = [Individual.next_trait_id + i for i in range(n_triangles)]
        Individual.next_trait_id += n_triangles
        return Individual(tl, cl, ids)

    @staticmethod
    def cross(a, b, average=False):
        """
        Cross the genomes of two parents to create a child. This will take the disjoint and excess genes from the most
        fit parent and randomly choose between the matching ones.

        Note: if the individuals have not been scored, then the first one will be treated as the most fit.
        Note: the render order of the most fit individual will be used.

        :param a: First parent.
        :param b: Second parent.
        :param average: If enabled, then matching gens are averaged instaed of randomly selected.
        :return: A child which is the result of crossing the individuals.
        """
        if a < b:  # a should be the most fit individual
            a, b = b, a

        a_ids = sorted(zip(a.trait_ids, range(len(a.trait_ids))))
        b_ids = sorted(zip(b.trait_ids, range(len(b.trait_ids))))

        n = len(a_ids)  # lists we can index into and assign items to
        c = Individual(list(range(n)), list(range(n)), list(range(n)))

        i, j = 0, 0
        while i < len(a_ids):
            if j >= len(b_ids) or a_ids[i][0] < b_ids[j][0]:
                # excess from a or disjoint from a, include
                k = a_ids[i][1]
                c[k] = deepcopy(a[k])
                i += 1
            elif a_ids[i][0] > b_ids[j][0]:
                # disjoint from b, ignore
                j += 1
            elif average:  # same trait id and average
                k = a_ids[i][1]
                l = b_ids[j][1]
                triangles = (a[k][0] + b[l][0]) / 2
                color = (a[k][1] + b[l][1]) / 2
                c[k] = (np.int32(triangles), np.uint8(color), a_ids[i][0])
                i += 1
                j += 1
            else:  # same trait id and not average
                k = a_ids[i][1]
                if random.rand() < 0.5:
                    c[k] = deepcopy(a[k])
                else:
                    l = b_ids[j][1]
                    c[k] = deepcopy(b[l])

                i += 1
                j += 1
        return c

    @staticmethod
    def compatibility_distance(a, b):
        """
        Compute the compatibility distance function δ. The value represents how different the two individuals are.
        :param a: First individual
        :param b: Second individual
        :return: The compatibility distance.
        """
        # implement a semi-dumb version for now which is hopefully accurate enough (optimal would be SSIM^2)
        # based on the NEAT algorithm's compatibility distance metric

        # get ids and their indexes sorted by trait
        a_ids = sorted(zip(a.trait_ids, range(len(a.trait_ids))))
        b_ids = sorted(zip(b.trait_ids, range(len(b.trait_ids))))

        i, j = 0, 0
        matching = 0
        disjoint = 0
        matching_diff = []
        while i < len(a_ids) and j < len(b_ids):
            if a_ids[i][0] == b_ids[j][0]:
                matching += 1
                # now sum what is basically a normalized euclidean distance
                k, l = a_ids[i][1], b_ids[j][1]
                matching_diff.append((
                    np.linalg.norm(np.float64(a.triangles[k] - b.triangles[l]) / resolution) +
                    np.linalg.norm(np.float64(a.colors[k] - b.colors[l]) / 256)
                ) / 2)
                i += 1
                j += 1

            elif a_ids[i][0] < b_ids[j][0]:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1

        matching_diff = 0 if not matching else np.sum(matching_diff) / matching
        excess = len(a_ids) - i + len(b_ids) - j
        n = disjoint + excess + matching
        distance = matching_diff * DIST_MATCHING_DIFF_COST +\
                   (excess / n) * DIST_EXCESS_COST +\
                   (disjoint / n) * DIST_DISJOINT_COST
        return distance

    def __matmul__(self, other):
        return Individual.cross(self, other)

    def __mod__(self, other):
        return Individual.compatibility_distance(self, other)

    def mutate(self):
        # Mutation types:
        # 1) remove a triangle
        # 2) shift shade of a triangle
        # 3) shift points of a triangle
        # 4) add a triangle
        # 5) swap two triangles (which one renders over the others)
        while len(self) > 1 and random.rand() < PROB_DEL_TRI:
            choice = random.randint(0, len(self))
            del self[choice]

        for i in range(len(self)):
            if random.rand() < PROB_MUT_TRI:
                for v in range(3):
                    self.triangles[i][v, 0] = \
                        (self.triangles[i][v, 0] + random.randn() * MUT_TRI_VERT_MEAN * resolution[0])\
                        .clip(0, resolution[0])
                    self.triangles[i][v, 1] = \
                        (self.triangles[i][v, 1] + random.randn() * MUT_TRI_VERT_MEAN * resolution[1])\
                        .clip(0, resolution[1])

            if random.rand() < PROB_MUT_TRI:
                self.colors[i] = np.uint8((self.colors[i] + random.randn() * MUT_TRI_COLOR_MEAN).clip(0, 255))

        while random.rand() < PROB_ADD_TRI:
            v, c = randomly_generate_triangle()
            self.append((v, c, Individual.next_trait_id))
            Individual.next_trait_id += 1

        while len(self) > 1 and random.rand() < PROB_ALTER_ORDERING:
            i = random.randint(0, len(self) - 1)
            self[i], self[i+1] = self[i+1], self[i]

    def render(self):
        return calculate_image(self.triangles, self.colors)

    def append(self, pair):
        self.triangles.append(pair[0])
        self.colors.append(pair[1])
        self.trait_ids.append(pair[2])

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __getitem__(self, item):
        return self.triangles[item], self.colors[item], self.trait_ids[item]

    def __setitem__(self, key, value):
        self.triangles[key] = value[0]
        self.colors[key] = value[1]
        self.trait_ids[key] = value[2]

    def __delitem__(self, key):
        del self.triangles[key]
        del self.colors[key]
        del self.trait_ids[key]

    def __len__(self):
        assert len(self.triangles) == len(self.colors) == len(self.trait_ids)
        return len(self.triangles)

Individual.next_trait_id = 1


def randomly_generate_triangle():
    return np.int32(random.rand(3, 2) * resolution), np.uint8(random.rand(4) * 256)


def randomly_generate_population(pop_size, avg_triangles):
    return [
        Individual.generate(np.random.randint(min(1, avg_triangles // 2), avg_triangles * 2))
        for _ in range(pop_size)
    ]


def _calculate_ssim(tfsess, b):
    # requires init_tensors first
    # b is a batch of BATCH_SIZE images
    ssim_v, summary = tfsess.run([_calculate_ssim.tensor, _calculate_ssim.summary], feed_dict={_calculate_ssim.im_b: b})
    tf_log_writer.add_summary(summary, generation)
    return ssim_v


def sort_two_lists(a, b, reverse=False):
    assert len(a) == len(b)
    indexes = list(range(len(a)))
    indexes.sort(key=a.__getitem__, reverse=reverse)
    return list(map(a.__getitem__, indexes)), list(map(b.__getitem__, indexes))


def train(tfsess, target_ssim=0.7, intermediate_path=None):
    global generation
    population = randomly_generate_population(GENERATION_SIZE, INIT_AVERAGE_TRIANGLES)
    generation = 0
    while True:
        # population_fitness = []
        for b in range(GENERATION_SIZE // BATCH_SIZE):
            image_batch = [population[i+b*BATCH_SIZE].render() for i in range(BATCH_SIZE)]
            fitness_scores = _calculate_ssim(tfsess, image_batch)
            for i in range(BATCH_SIZE):
                population[i+b*BATCH_SIZE].fitness = fitness_scores[i]
            del image_batch, fitness_scores

        if intermediate_path:
            imageio.imwrite('{}/gen{}.png'.format(intermediate_path, generation), np.uint8(population[0].render() * 256))

        population.sort(reverse=True)
        print("Gen {}; Best: {}.".format(generation, population[0].fitness))

        # check if we have reached our target_loss
        if generation >= MAX_GENERATIONS:
            break  # Not an optimal solution, but we have to stop somewhere
        elif population[0].fitness >= target_ssim:
            break

        # drop the lowest pops
        population = population[:GENERATION_CARRYOVER]

        # duplicate the top members to fill the gap
        n_babies = GENERATION_SIZE - GENERATION_CARRYOVER
        while n_babies > 0:
            i, j = random.randint(0, GENERATION_CARRYOVER, 2)
            if i == j:
                continue
            elif population[i] % population[j] < MAX_DISTANCE_FOR_BREEDING or random.rand() < INTERSPECIES_BREEDING_RATE:
                child = population[i] @ population[j]
            elif random.rand() < 0.2:  # trapdoor to prevent a forever loop if few or none can breed with each-other
                child = deepcopy(population[i])
            else:  # try a new pair
                continue

            child.mutate()
            population.append(child)
            n_babies -= 1

        # randomly mutate all individuals
        for i in population[GENERATION_CARRYOVER:]:
            i.mutate()

        generation += 1

    return population[0]


def init_tensors(tfsess, image):
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
    global resolution
    image = imageio.imread(photo)
    resolution = (image.shape[1], image.shape[0]) # because we store in Row-major order

    renderer = Renderer(resolution)

    with tf.Session() as sess:
        init_tensors(sess, image)
        best_indv = train(sess, target_ssim=target_ssim, intermediate_path=output if save_intermediate else None)
        imageio.imwrite('{}/final.png'.format(output), np.uint8(best_indv.render() * 256.0))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Use a genetic algorithm to create a represetnation of an image.")
    # parser.add_argument("source", metavar='src', type=str, help="source image")
    # parser.add_argument("output", metavar='dst', type=str, help="output destination")
    # parser.add_argument()
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] in ['True', 'False'])
