layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 tex;
    vec3 tangent;
};

Vertex[36] cubeVertices = Vertex[](
    // Back face
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, -0.5),
        vec3(0.0, 0.0, -1.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    // Front face
    Vertex(
        vec3(-0.5, -0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, 0.5),
        vec3(0.0, 0.0, 1.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    // Left face
    Vertex(
        vec3(-0.5, 0.5, 0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, -0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, 0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, 0.5),
        vec3(-1.0, 0.0, 0.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    // Right face
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, -0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, -0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(1.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, -0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(0.0, 1.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(1.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, 0.5),
        vec3(1.0, 0.0, 0.0),
        vec2(0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    ),
    // Bottom face
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(0.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, -0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(1.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, 0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(1.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, -0.5, 0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(1.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, 0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(0.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, -0.5, -0.5),
        vec3(0.0, -1.0, 0.0),
        vec2(0.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    // Top face
    Vertex(
        vec3(-0.5, 0.5, -0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(0.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(1.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, -0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(1.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(0.5, 0.5, 0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(1.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, -0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(0.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    ),
    Vertex(
        vec3(-0.5, 0.5, 0.5),
        vec3(0.0, 1.0, 0.0),
        vec2(0.0, 0.0),
        vec3(1.0, 0.0, 0.0)
    )
);

out vec3 localPos;

uniform mat4 projection;
uniform mat4 view;

void main() {
    mat4 transform = projection * view;

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            Vertex v = cubeVertices[i * 6 + j];
            localPos = v.position;
            gl_Position = transform * vec4(v.position, 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}
