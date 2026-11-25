#version 460

// no_api's examples/res/shader.slang, f_main (kgp_noapi.c). No root pointer.

layout(location = 0) in vec2 uvs;
layout(location = 1) in vec3 col;
layout(location = 0) out vec4 outColor;

void main()
{
   outColor = vec4(col, 1.0);
}
