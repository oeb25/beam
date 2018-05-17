layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 5) in mat4 aModel;

out VS_OUT {
    vec3 Normal;
} vs_out;

uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 pos = aModel * vec4(aPos, 1.0);
    gl_Position = projection * view * pos;
    mat3 normalMatrix = mat3(transpose(inverse(view * aModel)));
    vs_out.Normal = normalize(vec3(projection * vec4(normalMatrix * aNormal, 0.0)));
}
