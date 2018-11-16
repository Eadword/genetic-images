from OpenGL.GL import *
from OpenGL.GL import shaders

import glfw
import numpy as np
import os

_INSTANCE = None

vertex_data = np.array([0.75, 0.75,
                        0.75, -0.75,
                        -0.75, -0.75], dtype=np.float32)

color_data = np.array([1, 0, 0, 1,
                        0, 1, 0, 1,
                        0, 0, 1, 1], dtype=np.float32)


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

        glClearColor(0, 0, 0.4, 1)

        with open("shaders/vertex.glsl") as vs:
            vertex_shader = shaders.compileShader(vs.read(), GL_VERTEX_SHADER)
        with open("shaders/fragment.glsl") as fs:
            fragment_shader = shaders.compileShader(fs.read(), GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        self.vao_id = self._get_triangle_vao()

    def __del__(self):
        global _INSTANCE

        glfw.terminate()
        _INSTANCE = None

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        with self.shader:
            glBindVertexArray(self.vao_id)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _get_triangle_vao(self):
        # it is a container for buffers
        vao_id = glGenVertexArrays(1)

        glBindVertexArray(vao_id)
        vbo_id = glGenBuffers(2)

        # bind some GL_ARRAY_BUFFER to generated one id
        # it's a position buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id[0])
        # fill it with values
        glBufferData(GL_ARRAY_BUFFER, vertex_data, GL_STATIC_DRAW)
        # tell, how to interpret it
        glVertexAttribPointer(glGetAttribLocation(self.shader, 'vin_position'), 2, GL_FLOAT, GL_FALSE, 0, None)
        # open the valve, let it to be used.
        glEnableVertexAttribArray(0)

        # repeat it for colors.
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id[1])
        glBufferData(GL_ARRAY_BUFFER, color_data, GL_STATIC_DRAW)
        glVertexAttribPointer(glGetAttribLocation(self.shader, 'vin_color'), 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        # there we unbind current buffer and vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        # will bind VAO's at every draw action.
        return vao_id
