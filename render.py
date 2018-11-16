from OpenGL.GL import *
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
        self.shader = Shader(vertpath="shaders/vertex.glsl", fragpath="shaders/fragment.glsl")
        self.vao_id = self._get_triangle_vao()

    def __del__(self):
        global _INSTANCE

        glfw.terminate()
        _INSTANCE = None

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shader.use()
        glBindVertexArray(self.vao_id)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        self.shader.unbind()
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
        glVertexAttribPointer(self.shader.attrib_location('vin_position'), 2, GL_FLOAT, GL_FALSE, 0, None)
        # open the valve, let it to be used.
        glEnableVertexAttribArray(0)

        # repeat it for colors.
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id[1])
        glBufferData(GL_ARRAY_BUFFER, color_data, GL_STATIC_DRAW)
        glVertexAttribPointer(self.shader.attrib_location('vin_color'), 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        # there we unbind current buffer and vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        # will bind VAO's at every draw action.
        return vao_id


class Shader:
    def __init__(self, vertpath="", fragpath=""):
        self.id = 0
        self.vertpath = vertpath
        self.fragpath = fragpath

        if not os.path.exists(self.vertpath):
            raise FileNotFoundError("Vertex shader file ({}) doesn't exist!".format(self.vertpath))
        if not os.path.exists(self.fragpath):
            raise FileNotFoundError("Fragment shader file ({}) doesn't exist!".format(self.fragpath))

        # create unique shader program id
        self.id = glCreateProgram()

        # load and compile individual shaders
        vertsource = self._load_shader(self.vertpath)
        fragsource = self._load_shader(self.fragpath)
        vert_id = self._get_shader(vertsource, GL_VERTEX_SHADER)
        frag_id = self._get_shader(fragsource, GL_FRAGMENT_SHADER)

        # if it's ok, attach them to shader program
        glAttachShader(self.id, vert_id)
        glAttachShader(self.id, frag_id)

        # link program means make program obj with created executables for different programmable processors for
        # shaders, that were attached.
        glLinkProgram(self.id)
        # if something went wrong
        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.id)
            glDeleteProgram(self.id)
            # they should be deleted anyway
            glDeleteShader(vert_id)
            glDeleteShader(frag_id)
            raise RuntimeError("Error in program linking: " + info)

        # shaders are attached, program is linked -> full shader program with compiled executables is ready,
        # no need in individual shaders ids, i suppose
        glDeleteShader(vert_id)
        glDeleteShader(frag_id)

    def attrib_location(self, name):
        return glGetAttribLocation(self.id, name)

    def uniform_location(self, name):
        return glGetUniformLocation(self.id, name)

    def use(self):
        glUseProgram(self.id)

    @staticmethod
    def unbind():
        glUseProgram(0)

    @staticmethod
    def _get_shader(shader_source, shader_type):
        shader_id = 0
        try:
            shader_id = glCreateShader(shader_type)
            glShaderSource(shader_id, shader_source)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                raise RuntimeError('Shader compilation failed:\n %s'%info)
            return shader_id
        except:
            glDeleteShader(shader_id)
            raise

    @staticmethod
    def _load_shader(path):
        source_file = open(path)
        shader_source = source_file.read()
        source_file.close()
        return shader_source

    def __del__(self):
        glDeleteProgram(self.id)

    def __enter__(self):
        self.use()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unbind()
