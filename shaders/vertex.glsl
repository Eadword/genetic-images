#version 330 core

in  vec2 vin_position;
in  vec4 vin_color;
out vec4 vout_color;

void main(void) {
    vout_color = vin_color;
    gl_Position = vec4(vin_position, 1.0, 1.0);
}