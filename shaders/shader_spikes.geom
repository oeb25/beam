layout (triangles) in;
layout (triangle_strip, max_vertices = 9) out;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} gs_in[];

out GS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} gs_out;

uniform float time;

const float MAGINITUDE = 0.1;

float sin01(float t) {
    return sin(t) / 2.0 + 0.5;
}

vec4 explode(vec4 position, vec3 normal) {
    return position + vec4(normal, 0.0) * MAGINITUDE * sin01(time);
}

void emit(int i) {
    gl_Position = gl_in[i].gl_Position;
    gs_out.Normal = gs_in[i].Normal;
    gs_out.FragPos = gs_in[i].FragPos;
    gs_out.OriginalPos = gs_in[i].OriginalPos;
    gs_out.TBN = gs_in[i].TBN;
    gs_out.TexCoords = gs_in[i].TexCoords;
    EmitVertex();
}

void mid(vec3 surface_normal) {
#define avg(x) ((gs_in[0].x + gs_in[1].x + gs_in[2].x) / 3.0)

    vec4 mid = (gl_in[0].gl_Position + gl_in[1].gl_Position + gl_in[2].gl_Position) / 3.0;

    gl_Position = explode(mid, surface_normal);
    gs_out.Normal = avg(Normal);
    gs_out.FragPos = explode(vec4(avg(FragPos), 0.0), surface_normal).xyz;
    gs_out.OriginalPos = avg(OriginalPos);
    gs_out.TBN = avg(TBN);
    gs_out.TexCoords = avg(TexCoords);
    EmitVertex();
}

void main() {
    vec3 a = vec3(gl_in[0].gl_Position) - vec3(gl_in[1].gl_Position);
    vec3 b = vec3(gl_in[2].gl_Position) - vec3(gl_in[1].gl_Position);
    vec3 surface_normal = normalize(cross(a, b));

    emit(0);
    emit(1);
    mid(surface_normal);

    emit(1);
    emit(2);
    mid(surface_normal);

    emit(2);
    emit(0);
    mid(surface_normal);
}
