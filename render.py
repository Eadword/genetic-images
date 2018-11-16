from OpenGL.GL import *
from OpenGL.GL import shaders

import glfw
import numpy as np
import os

_INSTANCE = None

vertex_data = np.array([[200, 40], [100, 300], [800, 500],
                        [700, 40], [200, 500], [700, 500]], dtype=np.int32)

color_data = np.array([[255, 0, 0, 255], [255, 0, 0, 255], [255, 0, 0, 255],
                       [0, 255, 0, 255], [0, 255, 0, 80], [0, 255, 0, 80]], dtype=np.uint8)


class Renderer:
    def __new__(cls, *args, **kwargs):
        global _INSTANCE

        if not _INSTANCE:
            _INSTANCE = super(Renderer, cls).__new__(cls)
            _INSTANCE._run_init = True
        return _INSTANCE

    def __init__(self, resolution=(800,600), hidden=True):
        if self._run_init:
            self._run_init = False
        else:
            return

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        self.resolution = resolution
        self.vertex_buffer = None
        self.color_buffer = None

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(resolution[0], resolution[1], "Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW context.")
        glfw.make_context_current(self.window)

        if hidden:
            glfw.hide_window(self.window)

        glClearColor(0, 0, 0, 1)
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        with open("shaders/vertex.glsl") as vs:
            vertex_shader = shaders.compileShader(vs.read(), GL_VERTEX_SHADER)
        with open("shaders/fragment.glsl") as fs:
            fragment_shader = shaders.compileShader(fs.read(), GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        with self.shader:
            uloc = glGetUniformLocation(self.shader, 'screen_to_standard_transform')
            T = self._screen_to_standard_transform()
            glUniformMatrix3fv(uloc, 1, True, T)

        self.vao_id = self._get_triangle_vao()

    def __del__(self):
        global _INSTANCE

        glfw.terminate()
        _INSTANCE = None

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT)

        with self.shader:
            glBindVertexArray(self.vao_id)
            glDrawArrays(GL_TRIANGLES, 0, vertex_data.shape[0])
            glBindVertexArray(0)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _get_triangle_vao(self):
        # it is a container for buffers
        vao_id = glGenVertexArrays(1)

        glBindVertexArray(vao_id)
        vbo_id = glGenBuffers(2)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_id[0])
        glBufferData(GL_ARRAY_BUFFER, vertex_data, GL_STATIC_DRAW)
        glVertexAttribPointer(glGetAttribLocation(self.shader, 'vin_position'), 2, GL_INT, False, 0, None)
        glEnableVertexAttribArray(0)

        # repeat it for colors.
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id[1])
        glBufferData(GL_ARRAY_BUFFER, color_data, GL_STATIC_DRAW)
        glVertexAttribPointer(glGetAttribLocation(self.shader, 'vin_color'), 4, GL_UNSIGNED_BYTE, False, 0, None)
        glEnableVertexAttribArray(1)

        # there we unbind current buffer and vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao_id

    def _screen_to_standard_transform(self):
        """
        Transform from the 0,0 at top left going from (0 <= x =< xmax) and (0 <= y =< ymax)
        to (0,0) in the center going from (-1 =< x =< 1) and (0 =< y =< 1).
        Because it is not a linear transformation we need homogeneous coordinates, first a flip and scaling,
        then translate the u,v origin at the top left to the i,j origin in the center.
        """
        r = np.float32(self.resolution)

        # scale and flip
        # [
        #     [2.0/r[0],         0, 0],
        #     [       0, -2.0/r[1], 0],
        #     [       0,         0, 1]
        #  ]

        # translate to origin
        # [
        #     [1, 0, -1],
        #     [0, 1,  1],
        #     [0, 0,  1]
        # ]

        return np.array([
            [2.0/r[0],         0, -1],
            [       0, -2.0/r[1],  1],
            [       0,         0,  1]
        ], dtype=np.float32)
