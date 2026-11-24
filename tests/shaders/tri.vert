#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Vertex pulling: there are no vertex buffers in KoreGP. The vertex stage's
// root pointer (push-constant offset 0) leads to the vertex array; instances
// are shifted along x.

struct Vertex {
   vec2 pos;
   vec4 color;
};

layout(buffer_reference, scalar) readonly buffer VertexArray { Vertex v[]; };

layout(buffer_reference, scalar) readonly buffer TriVsRoot {
   VertexArray vertices;
   float instanceStep;
   float z;
};

layout(push_constant, scalar) uniform Push { TriVsRoot root; };

layout(location = 0) out vec4 color;

void main()
{
   Vertex v = root.vertices.v[gl_VertexIndex];
   vec2 pos = v.pos + vec2(root.instanceStep * float(gl_InstanceIndex), 0.0);
   gl_Position = vec4(pos, root.z, 1.0);
   color = v.color;
}
