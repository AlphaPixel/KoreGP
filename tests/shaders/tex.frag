#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

// A pixel shader reading the texture heap: each pixel takes the texel at
// its own coordinates from the texture its root names.

layout(set = 0, binding = 0) uniform texture2D kgpTextures[];

layout(buffer_reference, scalar) readonly buffer TexPsRoot {
   uint textureIndex;
};

layout(push_constant, scalar) uniform Push {
   layout(offset = 8) TexPsRoot root;
};

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 outColor;

void main()
{
   outColor = texelFetch(kgpTextures[root.textureIndex],
                         ivec2(gl_FragCoord.xy), 0) * color;
}
