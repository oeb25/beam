layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 5) in mat4 aModel;

out VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} vs_out;

uniform mat4 view;
uniform mat4 projection;

void main() {
#if 1
    vs_out.OriginalPos = aPos;
    vec4 pos = aModel * vec4(aPos, 1.0);
    vs_out.FragPos = vec3(pos);
    mat3 normalMatrix = transpose(inverse(mat3(aModel)));
    vs_out.Normal = normalMatrix * aNormal;
    vs_out.TexCoords = vec2(aTexCoords.x, vec2(1.0) - aTexCoords.y);

    vec3 T = normalize(normalMatrix * aTangent);
    vec3 N = normalize(normalMatrix * aNormal);

    T = normalize(T - dot(T, N) * N);

    vec3 B = cross(N, T);

    vs_out.TBN = mat3(T, B, N);;

    gl_Position = projection * view * pos;
#else
    vs_out.OriginalPos = aPos;
    vec4 pos = vec4(aPos, 1.0);
    vs_out.FragPos = vec3(pos);
    vs_out.Normal = aNormal;
    vs_out.TexCoords = aTexCoords;

    vec3 T = aTangent;
    vec3 N = aNormal;

    vec3 B = T;

    vs_out.TBN = mat3(T, B, N);;

    gl_Position = pos;
#endif
}
