#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Must be REJECTED: a pixel shader's root pointer is at push-constant
// offset 8; offset 0 belongs to the vertex stage.

layout(buffer_reference, scalar) readonly buffer Root { vec4 tint; };

layout(push_constant, scalar) uniform Push { Root root; };

layout(location = 0) out vec4 outColor;

void main()
{
   outColor = root.tint;
}
