layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

out vec2 TexCoords;

void e(vec2 p) {
    TexCoords = p;
    gl_Position = vec4((p - vec2(0.5)) * 2, 0.0, 1.0);
    EmitVertex();
}

void main() {
    const vec2 conors[] = vec2[](
        vec2(0, 0),
        vec2(1, 0),
        vec2(1, 1),
        vec2(0, 1)
    );

    e(conors[0]);
    e(conors[1]);
    e(conors[2]);
    EndPrimitive();

    e(conors[0]);
    e(conors[2]);
    e(conors[3]);
    EndPrimitive();
}
