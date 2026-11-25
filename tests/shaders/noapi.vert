#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// no_api's examples/res/shader.slang, v_main, in GLSL (kgp_noapi.c). There
// the push constant is `Vertex* data`; here the root pointer is the same
// pointer. The original's printf is left out.

struct Vertex {
   vec3 position;
   vec3 col;
   vec2 uvs;
};

layout(buffer_reference, scalar) readonly buffer VertexArray { Vertex v[]; };

layout(push_constant, scalar) uniform Push { VertexArray data; };

layout(location = 0) out vec2 uvs;
layout(location = 1) out vec3 col;

void main()
{
   Vertex v = data.v[gl_VertexIndex];
   gl_Position = vec4(v.position, 1.0);
   uvs = v.uvs;
   col = v.col;
}
