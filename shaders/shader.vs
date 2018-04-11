#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in mat4 aModel;

out VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
    mat4 Model;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vs_out.OriginalPos = aPos;
    mat4 model_ = aModel * model;
    vec4 pos = model_ * vec4(aPos, 1.0);
    vs_out.FragPos = vec3(pos);
    vs_out.Normal =  aNormal;
    vs_out.TexCoords = vec2(aTexCoords.x, -aTexCoords.y);
    vs_out.Model = model_;

    vec3 T = normalize(vec3(model_ * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model_ * vec4(aNormal, 0.0)));

    T = normalize(T - dot(T, N) * N);

    vec3 B = cross(N, T);

    vs_out.TBN = mat3(T, B, N);

    gl_Position = projection * view * pos;
}
