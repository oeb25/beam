#version 410 core

// out vec4 FragColor;

layout (location = 0) out vec3 aPosition;
layout (location = 1) out vec3 aNormal;
layout (location = 2) out vec4 aAlbedoSpec;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} fs_in;

uniform sampler2D tex_diffuse1;
uniform sampler2D tex_specular1;
uniform sampler2D tex_normal1;
uniform bool useNormalMap;
uniform vec3 viewPos;
uniform samplerCube skybox;

void main() {
    vec3 norm;
    if (useNormalMap) {
        norm = texture(tex_normal1, fs_in.TexCoords).rgb;
        norm = normalize(norm * 2.0 - 1.0);
        norm = normalize(fs_in.TBN * norm);
    } else {
        norm = fs_in.Normal;
    }

    aPosition = fs_in.FragPos;
    aNormal = norm;
    aAlbedoSpec.rgb = texture(tex_diffuse1, fs_in.TexCoords).rgb;
    aAlbedoSpec.a = length(texture(tex_specular1, fs_in.TexCoords).rgb);
}
