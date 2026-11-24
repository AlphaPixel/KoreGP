#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// The pixel stage's root pointer is the second one: push-constant offset 8.

layout(buffer_reference, scalar) readonly buffer TriPsRoot {
   vec4 tint;
};

layout(push_constant, scalar) uniform Push {
   layout(offset = 8) TriPsRoot root;
};

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 outColor;

void main()
{
   outColor = color * root.tint;
}
