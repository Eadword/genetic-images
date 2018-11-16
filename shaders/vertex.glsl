#version 330 core

in  vec2 vin_position;
in  vec4 vin_color;
out vec4 vout_color;

uniform mat3 screen_to_standard_transform;

void main(void) {
    vout_color = vin_color / 255.0;
    vec3 fpos_h = screen_to_standard_transform * vec3(vin_position, 1.0);

    gl_Position = vec4(fpos_h.xy, 1.0, fpos_h.z);
}