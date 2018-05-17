layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;
layout (location = 5) in mat4 aModel;

out VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vs_out.OriginalPos = aPos;
    mat4 model_ = aModel * model;
    vec4 pos = model_ * vec4(aPos, 1.0);
    vs_out.FragPos = vec3(pos);
    mat3 normalMatrix = transpose(inverse(mat3(model_)));
    vs_out.Normal = normalMatrix * aNormal;
    vs_out.TexCoords = vec2(aTexCoords.x, -aTexCoords.y);

#if 1
    vec3 T = normalize(vec3(model_ * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model_ * vec4(aNormal, 0.0)));

    T = normalize(T - dot(T, N) * N);

    vec3 B = cross(N, T);

    vs_out.TBN = mat3(T, B, N);
#else
    vec3 T = normalize(vec3(model_ * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model_ * vec4(aNormal, 0.0)));
    vec3 B = normalize(vec3(model_ * vec4(aBitangent, 0.0)));

    vs_out.TBN = mat3(T, B, N);
#endif

    gl_Position = projection * view * pos;
}
