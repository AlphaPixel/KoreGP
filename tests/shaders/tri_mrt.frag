#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// The "mrt" variant's pixel shader (tri_scene.h): tri.frag's colour to the
// first colour target and its complement to the second.

layout(buffer_reference, scalar) readonly buffer TriPsRoot {
   vec4 tint;
};

layout(push_constant, scalar) uniform Push {
   layout(offset = 8) TriPsRoot root;
};

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outComplement;

void main()
{
   outColor = color * root.tint;
   outComplement = vec4(1.0 - outColor.rgb, 1.0);
}
