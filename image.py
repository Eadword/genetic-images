import numpy as np
import tensorflow as tf


def init_tensors(resolution):
    global canvas, triangle_coordinates, triangle_color, locations_to_check
    canvas = tf.placeholder(np.uint8, shape=(resolution[0], resolution[1], 3))
    triangle_coordinates = tf.placeholder(np.int32, shape=(3, 2))
    triangle_color = tf.placeholder(np.uint8, shape=(4,))
    locations_to_check = tf.placeholder(np.int32, shape=(None, 2))

    global within_tensor, outside_tensor
    within_tensor = _in_triangle_tensor()
    outside_tensor = tf.logical_not(within_tensor)  # need inverse for numpy masking


def in_triangle(coords, loc):
    # http://blackpawn.com/texts/pointinpoly/
    coords = np.int32(coords)
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
        min_x = t_coords[:,0].min()
        min_y = t_coords[:,1].min()
        max_x = t_coords[:,0].max()
        max_y = t_coords[:,1].max()

        src_a = np.float64(t_color[3]) / 255

        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if not in_triangle(t_coords, np.array([x, y])):
                    continue

                # https://en.wikipedia.org/wiki/Alpha_compositing#Examples
                # simplified because the destination alpha is always 1
                im[x,y] = src_a * t_color[:3] + (1.0 - src_a) * im[x, y]
    return im


def calculate_image_with_tensors(tfsess, pop, resolution):
    im = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)

    for (t_coords, t_color) in pop:
        min_x = t_coords[:, 0].min()
        min_y = t_coords[:, 1].min()
        max_x = t_coords[:, 0].max()
        max_y = t_coords[:, 1].max()

        locations = np.indices((max_x-min_x,max_y-min_y), dtype=np.int32).swapaxes(0,1).swapaxes(1,2).reshape(-1,2) + [min_x, min_y]
        outside = tfsess.run(outside_tensor, feed_dict={triangle_coordinates: t_coords, locations_to_check: locations})
        locations = np.ma.array(locations, mask=np.column_stack((outside, outside))).compressed().reshape(-1,2)
        x_cords = locations[:,0]
        y_cords = locations[:,1]

        src_a = np.float64(t_color[3]) / 255
        src_c = t_color

        im[x_cords, y_cords] = src_a * src_c[:3] + im[x_cords, y_cords] * (1.0 - src_a)

    return im


def _in_triangle_tensor():
    coords = triangle_coordinates
    locs = locations_to_check

    v0 = coords[2] - coords[0]
    v1 = coords[1] - coords[0]
    v2 = locs - coords[0]
    dot00 = tf.tensordot(v0, v0, 1)
    dot01 = tf.tensordot(v0, v1, 1)
    dot02 = tf.tensordot(v0, v2, [[0], [1]])
    dot11 = tf.tensordot(v1, v1, 1)
    dot12 = tf.tensordot(v1, v2, [[0], [1]])
    denom = tf.cast(dot00 * dot11 - dot01 * dot01, np.float64)
    inv_denom = tf.pow(denom, -1)
    u = tf.cast(dot11 * dot02 - dot01 * dot12, np.float64) * inv_denom
    v = tf.cast(dot00 * dot12 - dot01 * dot02, np.float64) * inv_denom

    a = tf.greater_equal(u, 0)
    b = tf.greater_equal(v, 0)
    c = tf.less(u + v, 1)
    within = tf.logical_and(tf.logical_and(a, b), c)
    return within


# def _image_calculation_tensor():
#     t_coords = triangle_coordinates
#     t_color = triangle_color
#     image = canvas
#
#     min_x = t_coords[:,0].min()
#     min_y = t_coords[:,1].min()
#     max_x = t_coords[:,0].max()
#     max_y = t_coords[:,1].max()
#
#     # https://stackoverflow.com/questions/18359671/fastest-method-to-create-2d-numpy-array-whose-elements-are-in-range
#     x_locations, y_locations = tf.meshgrid(tf.range(min_x, max_x), tf.range(min_y, max_y))
#     locations = tf.transpose(tf.stack([x_locations, y_locations], axis=0))
#     locations_to_check
#     outside = outside_tensor