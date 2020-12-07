from image import calculate_image
from render import Renderer

import numpy as np
import numpy.random as random
import tensorflow.compat.v1 as tf

from datetime import datetime
import imageio
import sys
from copy import deepcopy, copy
import argparse

MAX_GENERATIONS = 1000000
GENERATION_SIZE = 60
SPECIES_COUNT = 1  # must evenly divide generation size
SPECIES_SIZE = GENERATION_SIZE // SPECIES_COUNT
SPECIES_NON_IMPROV_DEATH = 10  # kill a species if after x generations it has made no improvements
SPECIES_CARRYOVER = int(SPECIES_SIZE * 0.4)  # the percentage which become parents
KEEP_SPECIES_BEST = True # if true, keep the best of each species unaltered
CROSSOVER_AVERAGING_RATE = 0.2

INIT_AVERAGE_TRIANGLES = 5

DIST_MATCHING_DIFF_COST = 0.4
DIST_EXCESS_COST = 1.0
DIST_DISJOINT_COST = 1.0

PROB_MUTATE_INDV = 0.70  # chance an individual will be mutated (not guaranteed to be changed)
PROB_DEL_TRI = 0.105  # chance any triangle is deleted
PROB_ADD_TRI = 0.10  # change a random triangle is added
MAX_TRIANGLE_FACTOR = 0.02  # Allow no more more than x*num_pixels triangles

PROB_MUT_TRI = 0.6  # chance a triangle is mutated
MUT_TRI_VERT_MEAN = 0.02  # on average, move x*resolution
MUT_TRI_COLOR_MEAN = 6.0  # on average, change each color channel by x

PROB_ALTER_ORDERING = 0.30  # chance any two triangles are swapped thus changing render order

HIDDEN_RENDER=False

random.seed()

generation = 0
resolution = (0, 0)


class Species(list):
    # a list of individuals
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = 0
        self.last_improvement = 0
        self.id = Species.next_species_id
        Species.next_species_id += 1

    @staticmethod
    def generate(species_size, avg_triangles):
        # by mutating a single individual to comprise the whole group, they are actually related genetically which only
        # makes sense given we want to cross similar genetics
        alpha = Individual.generate(np.random.randint(min(1, avg_triangles // 2), avg_triangles * 2))
        s = Species([alpha])
        for _ in range(species_size - 1):
            beta = deepcopy(alpha)
            beta.mutate()
            s.append(beta)
        return s

    def evaluate(self, tfsess):
        images = [i.render() for i in self]
        fitness_scores = _calculate_inv_dist(tfsess, images)
        for i in range(len(self)):
            self[i].fitness = fitness_scores[i]

        self.sort(reverse=True)
        if self[0].fitness > self.fitness:
            self.fitness = self[0].fitness
            self.last_improvement = generation

    def next_generation(self):
        # drop the lowest pops
        parents = self[:SPECIES_CARRYOVER]
        if KEEP_SPECIES_BEST:
            best = self[0]
            self.clear()
            self.append(best)
        else:
            self.clear()
        self._fill_niche(parents)

    def split_species(self):
        # randomly choose two new alphas and then fill out their respective species
        i, j = 0, 0
        while i == j:
            i, j = random.randint(0, len(self), 2)

        # distance from all members to the alphas
        a_dist = [self[i] % m for m in self]
        b_dist = [self[j] % m for m in self]

        b = Species()
        for i, (ad, bd) in enumerate(zip(a_dist, b_dist)):
            if i == j or ad > bd:
                t = self[i - len(b)]  # i - len(b) to offset by the number of removed elements
                del self[i - len(b)]
                b.append(t)
            if ad <= bd:
                # stays in self
                continue

        self.id = Species.next_species_id
        Species.next_species_id += 1
        self.fitness = self[0].fitness
        self.last_improvement = generation
        self._fill_niche(self)

        b.fitness = b[0].fitness
        b.last_improvement = generation
        b._fill_niche(b)

        return b

    def _fill_niche(self, parents):
        n_babies = SPECIES_SIZE - len(self)
        assert len(parents) > 0
        if len(parents) == 1:
            parents = [parents[0], deepcopy(parents[0])]
            parents[1].mutate()

        while n_babies > 0:
            i, j = random.randint(0, len(parents), 2)
            if i == j:
                child = parents[i]
                child.mutate()
            else:
                child = parents[i] @ parents[j]
                if random.rand() < PROB_MUTATE_INDV:
                    child.mutate()
            self.append(child)
            n_babies -= 1

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

Species.next_species_id = 1


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
    def cross(a, b):
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
        average = random.rand() < CROSSOVER_AVERAGING_RATE

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
        # 2) shift a shade of a triangle
        # 3) shift a point of a triangle
        # 4) add a triangle
        # 5) swap two triangles (which one renders over the others)
        while len(self) > 1 and random.rand() < PROB_DEL_TRI:
            choice = random.randint(0, len(self))
            del self[choice]

        while random.rand() < PROB_MUT_TRI:
            i = random.randint(0, len(self))
            v = random.randint(0, 3)
            self.triangles[i][v, 0] = \
                (self.triangles[i][v, 0] + random.randn() * MUT_TRI_VERT_MEAN * resolution[0])\
                .clip(0, resolution[0])
            self.triangles[i][v, 1] = \
                (self.triangles[i][v, 1] + random.randn() * MUT_TRI_VERT_MEAN * resolution[1])\
                .clip(0, resolution[1])

        while random.rand() < PROB_MUT_TRI:
            i = random.randint(0, len(self))
            c = random.randint(0, 4)
            self.colors[i][c] = np.uint8((self.colors[i][c] + random.randn() * MUT_TRI_COLOR_MEAN).clip(0, 256))

        max_triangles = MAX_TRIANGLE_FACTOR*resolution[0]*resolution[1]
        while len(self) < max_triangles and random.rand() < PROB_ADD_TRI:
            v, c = randomly_generate_triangle()
            self.append((v, c, Individual.next_trait_id))
            Individual.next_trait_id += 1

        while len(self) > 1 and random.rand() < PROB_ALTER_ORDERING:
            i = random.randint(0, len(self) - 1)
            self[i], self[i+1] = self[i+1], self[i]

    def render(self, gen_image=True, render_to_window=False):
        renderer = Renderer()  # should be trivial assignment from locally stored instance
        renderer.data(
            self.triangles,
            np.tile(self.colors, (1, 3)).reshape((-1, 4))
        )
        return renderer.render(gen_image=gen_image, render_to_window=render_to_window)

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


def randomly_generate_triangle(transparent=False):
    A = np.int32(random.rand(2) * resolution)
    B = A + random.randn(2) * MUT_TRI_VERT_MEAN
    C = B + random.randn(2) * MUT_TRI_VERT_MEAN
    verts = np.int32([A, B, C])

    colors = np.uint8(random.rand(4) * 256)
    if transparent:
        return verts, colors * [1,1,1,0]
    else:
        return verts, colors


def _calculate_ssim(tfsess, i):
    # requires init_tensors first
    # i is a single image
    ssim_v, summary = tfsess.run([_calculate_ssim.tensor, _calculate_ssim.summary], feed_dict={_calculate_ssim.im_b: i})
    tf_log_writer.add_summary(summary, generation)
    return ssim_v


def _calculate_inv_dist(tfsess, b):
    # requires init_tensors first
    # b is a batch of SPECIES_COUNT images
    diff_v, summary = tfsess.run([_calculate_inv_dist.tensor, _calculate_inv_dist.summary], feed_dict={_calculate_inv_dist.im_b: b})
    tf_log_writer.add_summary(summary, generation)
    return diff_v


def simulate(tfsess, target_ssim=0.7, intermediate_path=None):
    global generation
    population = [Species.generate(SPECIES_SIZE, INIT_AVERAGE_TRIANGLES) for _ in range(SPECIES_COUNT)]
    generation = 0
    top_species = 0
    last_save_at_fitness = 0

    while True:
        for s in population:
            s.evaluate(tfsess)
        population.sort(reverse=True)
        if type(top_species) == set:
            # resolve the tuple type now that we know how they did.
            if population[0].id in top_species:
                top_species = population[0].id
            else:
                top_species = top_species.pop()
        if top_species != population[0].id:
            if top_species != 0:
                print("Species {} has overtaken {}!".format(population[0].id, top_species))
            top_species = population[0].id
            top_species_max_non_improv = SPECIES_NON_IMPROV_DEATH
        else:
            top_species_max_non_improv = max(generation - population[0].last_improvement, top_species_max_non_improv)

        image_of_champ = population[0][0].render(render_to_window=not HIDDEN_RENDER)
        _calculate_ssim(tfsess, image_of_champ)  # for logging to tensorboard

        if intermediate_path and population[0].fitness > last_save_at_fitness * 1.01:
            # if the best image is at least 1% better than the last time we saved it, write it to disk
            last_save_at_fitness = population[0].fitness
            imageio.imwrite('{}/gen{}.png'.format(intermediate_path, generation), np.uint8(image_of_champ * 256.0))

        print("Gen {}; Improvements: {}; Species Fitness: {}; Tricount: {}.".format(
            generation,
            sum([s.last_improvement == generation for s in population]),
            [(s.id, "{:.8f}".format(s.fitness)) for s in population],
            len(population[0][0])
        ))

        # check if we have reached our target_loss
        if generation >= MAX_GENERATIONS:
            break  # Not an optimal solution, but we have to stop somewhere
        elif population[0].fitness >= target_ssim:
            break

        for s in population:
            s.next_generation()

        new_population = [population[0]]  # don't allow best species to die
        new_population.extend(filter(lambda s: generation - s.last_improvement <= top_species_max_non_improv, population[1:]))
        if len(new_population) == 0:
            print("All species have died simultaneously.")
            return population[0]
        population = new_population
        survivors = len(population)
        while len(population) < SPECIES_COUNT:
            i = random.randint(0, survivors)
            population.append(population[i].split_species())
            if i == 0:
                # need to update our top species id to reflect this, so list both options for now
                if type(top_species) == int:
                    top_species = {population[0].id, population[-1].id}
                else:
                    top_species = top_species | {population[0].id, population[-1].id}

        generation += 1
    return population[0][0]


def tensor_summaries(tensor):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(tensor)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))

        max = tf.reduce_max(tensor)
        min = tf.reduce_min(tensor)
        mean_s = tf.summary.scalar('mean', mean)
        stddev_s = tf.summary.scalar('stddev', stddev)
        max_s = tf.summary.scalar('max', max)
        min_s = tf.summary.scalar('min', min)
        hs = tf.summary.histogram('values', tensor)
        return tf.summary.merge([mean_s, stddev_s, max_s, min_s, hs])


def init_tensors(tfsess, image):
    with tf.name_scope('base_image'):
        im_a = tf.constant(np.float32(image).reshape(1, resolution[1], resolution[0], 3) / 256.0)
        _calculate_ssim.im_a = im_a
        _calculate_inv_dist.im_a = im_a
    with tf.name_scope('input_images'):
        _calculate_ssim.im_b = tf.placeholder(np.float32, shape=(resolution[1], resolution[0], 3))
        _calculate_inv_dist.im_b = tf.placeholder(np.float32, shape=(SPECIES_SIZE, resolution[1], resolution[0], 3))
    with tf.name_scope('ssim'):
        _calculate_ssim.tensor = tf.image.ssim(im_a, tf.reshape(_calculate_ssim.im_b, (1, resolution[1], resolution[0], 3)), 1.0)[0]
        _calculate_ssim.summary = tf.summary.scalar('ssim_v', _calculate_ssim.tensor)
    with tf.name_scope('inv_dist'):
        _calculate_inv_dist.tensor = 1.0 / tf.linalg.norm(tf.reshape(im_a - _calculate_inv_dist.im_b, (SPECIES_SIZE, -1)), axis=1)
        _calculate_inv_dist.summary = tensor_summaries(_calculate_inv_dist.tensor)

    global tf_log_writer
    tf_log_writer = tf.summary.FileWriter('./log/{}'.format(datetime.now()), tfsess.graph)


def main(photo, output, target_ssim=0.7, save_intermediate=False):
    global resolution
    image = imageio.imread(photo)
    resolution = np.int32([image.shape[1], image.shape[0]]) # because we store in Row-major order

    renderer = Renderer(resolution, hidden=HIDDEN_RENDER)

    with tf.Session() as sess:
        init_tensors(sess, image)
        best_indv = simulate(sess, target_ssim=target_ssim, intermediate_path=output if save_intermediate else None)
        imageio.imwrite('{}/final.png'.format(output), np.uint8(best_indv.render() * 256.0))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Use a genetic algorithm to create a represetnation of an image.")
    # parser.add_argument("source", metavar='src', type=str, help="source image")
    # parser.add_argument("output", metavar='dst', type=str, help="output destination")
    # parser.add_argument()
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] in ['True', 'true'])
