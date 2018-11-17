# https://learnopengl.com/Advanced-OpenGL/Framebuffers

from OpenGL.GL import *
from OpenGL.GL import shaders

import glfw
import numpy as np

_INSTANCE = None


class Renderer:
    def __new__(cls, *args, **kwargs):
        global _INSTANCE

        if _INSTANCE is None:
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

        self.hidden = hidden
        self.resolution = resolution
        self.vao_id = 0
        self.vert_vbo_id = 0
        self.color_vbo_id = 0
        self.vertex_buffer = None
        self.color_buffer = None
        self.frame_buffer = None
        self.result_texture = None
        self.n_verts = 0

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.window = glfw.create_window(resolution[0], resolution[1], "Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW context.")
        glfw.make_context_current(self.window)

        if hidden:
            glfw.hide_window(self.window)

        self._init_alpha()
        self._init_frame_buffer_objects()

        with open("shaders/vertex.glsl") as vs:
            vertex_shader = shaders.compileShader(vs.read(), GL_VERTEX_SHADER)
        with open("shaders/fragment.glsl") as fs:
            fragment_shader = shaders.compileShader(fs.read(), GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex_shader, fragment_shader)
        with self.program:
            uloc = glGetUniformLocation(self.program, 'screen_to_standard_transform')
            T = self._screen_to_clip()
            glUniformMatrix3fv(uloc, 1, True, T)

        self.vao_id = self._init_vao()

    def __del__(self):
        global _INSTANCE
        # TODO: not sure if we can make a clean destructor, especially if python is shutting down
        # Come back to this if we are going to actually start and stop it
        # glDeleteVertexArrays(1, self.vao_id)
        # glDeleteBuffers(2, [self.vert_vbo_id, self.color_vbo_id])
        # glDeleteFramebuffers(self.frame_buffer)
        # glDeleteTextures(self.result_texture)
        # glDeleteProgram(self.program)
        glfw.terminate()

        _INSTANCE = None

    def data(self, vertices, colors):
        vertices = np.ascontiguousarray(vertices, dtype=np.int32)
        colors = np.ascontiguousarray(colors, dtype=np.uint8)

        glBindVertexArray(self.vao_id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vert_vbo_id)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STREAM_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo_id)
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STREAM_DRAW)

        # there we unbind current buffer and vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.n_verts = vertices.size // 2

    def render(self, gen_image=True):
        image = None

        glUseProgram(self.program)
        glBindVertexArray(self.vao_id)

        if gen_image:
            glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
            glClear(GL_COLOR_BUFFER_BIT)
            glDrawArrays(GL_TRIANGLES, 0, self.n_verts)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            image = glReadPixels(0,0,self.resolution[0], self.resolution[1], GL_RGB, GL_FLOAT)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            # numpy is row-major order and the screen col-major so the shapes appear to be wrong but are not
            image = np.fromstring(image, dtype=np.float32)\
                .reshape((self.resolution[1], self.resolution[0], 3))
            # OpenGL reads from bottom up, so we want to flip the y axis
            image = np.flip(image, axis=0)

        if not self.hidden:
            glClear(GL_COLOR_BUFFER_BIT)
            glDrawArrays(GL_TRIANGLES, 0, self.n_verts)
            glfw.swap_buffers(self.window)

        glUseProgram(0)
        glBindVertexArray(0)
        glfw.poll_events()

        return image

    def _init_vao(self):
        # it is a container for buffers
        vao_id = glGenVertexArrays(1)

        glBindVertexArray(vao_id)
        self.vert_vbo_id, self.color_vbo_id = glGenBuffers(2)

        glBindBuffer(GL_ARRAY_BUFFER, self.vert_vbo_id)
        glVertexAttribPointer(glGetAttribLocation(self.program, 'vin_position'), 2, GL_INT, False, 0, None)
        glEnableVertexAttribArray(0)

        # repeat it for colors.
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo_id)
        glVertexAttribPointer(glGetAttribLocation(self.program, 'vin_color'), 4, GL_UNSIGNED_BYTE, False, 0, None)
        glEnableVertexAttribArray(1)

        # there we unbind current buffer and vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao_id

    def _init_frame_buffer_objects(self):
        self.frame_buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        self.result_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.result_texture)

        self._init_alpha()

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.resolution[0], self.resolution[1], 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.result_texture, 0)

        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _init_alpha(self):
        glClearColor(0, 0, 0, 1)
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _screen_to_clip(self):
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
