import imageio
import numpy as np
import sys
import argparse

MAX_GENERATIONS = 100
GENERATION_SIZE = 100
INIT_POP_AVERAGE_TRIANGLES = 5
np.random.seed(908759798)

# A Pop is a list of Triangles
# A Triangle is a tuple of two arrays, the first it's three coordinates and the second its RGBA color


def randomly_generate_pop(triangles, resolution):
    return [
        (
            np.uint32(np.random.rand(3, 2) * resolution),
            np.uint8(np.random.rand(4) * 256)
        ) for _ in range(triangles)
    ]


def randomly_generate_population(pop_size, avg_triangles, resolution):
    return [
        randomly_generate_pop(
            np.random.randint(min(1, avg_triangles // 2), avg_triangles * 2),
            resolution
        )
        for _ in range(pop_size)
    ]


def calculate_loss(a, b):
    return np.sum(np.sqrt(a.astype(np.float64)**2 - b.astype(np.float64)**2))


def in_triangle(coords, loc):
    # http://blackpawn.com/texts/pointinpoly/
    v0 = coords[2] - coords[0]
    v1 = coords[1] - coords[0]
    v2 = loc - coords[0]
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1 / np.float64(dot00 * dot11 - dot01 * dot01)
    u = np.float64(dot11 * dot02 - dot01 * dot12) * inv_denom
    v = np.float64(dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v < 1)


def calculate_image(pop, resolution):
    im = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
    for (t_coords, t_color) in pop:
        min_x = t_coords[:,:1].min()
        min_y = t_coords[:,1:].min()
        max_x = t_coords[:,:1].max()
        max_y = t_coords[:,1:].max()

        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if not in_triangle(t_coords, np.array([x, y])):
                    continue

                # https://en.wikipedia.org/wiki/Alpha_compositing#Examples
                # simplified because the destination alpha is always 1
                src_a = np.float64(t_color[3]) / 255
                im[x,y] = src_a * t_color[:3] + (1.0 - src_a) * im[x,y,:3]
    return im


def calculate_population_fitness(image, population):
    return [
        1 / calculate_loss(image, calculate_image(pop, image.shape[:1]))
        for pop in population
    ]


def main(photo, output, target_loss=0.05, save_intermediate=False):
    image = imageio.imread(sys.argv[0])
    population = randomly_generate_population(GENERATION_SIZE, INIT_POP_AVERAGE_TRIANGLES, image.shape[:1])
    fitness = calculate_population_fitness(image, population)
    # sort population by fitness, drop lowest % and then mutate and breed the remaining


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Use a genetic algorithm to create a represetnation of an image.")
    # parser.add_argument("source", metavar='src', type=str, help="source image")
    # parser.add_argument("output", metavar='dst', type=str, help="output destination")
    # parser.add_argument()
    main(sys.argv[0], sys.argv[1], int(sys.argv[2]), sys.argv[3])