#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// The "mesh" variant's pixel shader: tri.frag with tri.mesh's per-primitive
// colour multiplied in. Vertex yellow x primitive yellow x tint = red, as
// in the plain scene; a per-primitive input that does not arrive is black.

layout(buffer_reference, scalar) readonly buffer TriPsRoot {
   vec4 tint;
};

layout(push_constant, scalar) uniform Push {
   layout(offset = 8) TriPsRoot root;
};

layout(location = 0) in vec4 color;
layout(location = 1) perprimitiveEXT in vec4 primColor;
layout(location = 0) out vec4 outColor;

void main()
{
   outColor = color * primColor * root.tint;
}
