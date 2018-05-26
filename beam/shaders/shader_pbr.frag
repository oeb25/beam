layout (location = 0) out vec3 aPosition;
layout (location = 1) out vec4 aNormal;
layout (location = 2) out vec4 aAlbedo;
layout (location = 3) out vec4 aMetallicRoughnessAo;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} fs_in;

uniform sampler2D tex_normal;
uniform sampler2D tex_albedo;
uniform sampler2D tex_metallic;
uniform sampler2D tex_roughness;
uniform sampler2D tex_ao;

uniform vec3 viewPos;

void main() {
    vec3 norm = texture(tex_normal, fs_in.TexCoords).rgb;
    norm = normalize(norm * 2.0 - 1.0);
    norm = normalize(fs_in.TBN * norm);
    // norm = fs_in.Normal;

    vec3 albedo = texture(tex_albedo, fs_in.TexCoords).rgb;
    float roughness = texture(tex_roughness, fs_in.TexCoords).r;
    float metallic = texture(tex_metallic, fs_in.TexCoords).r;
    float ao = texture(tex_ao, fs_in.TexCoords).r;

    aPosition = fs_in.FragPos;
    aNormal.rgb = norm;
    aAlbedo.rgb = albedo;
    aMetallicRoughnessAo.rgb = vec3(metallic, roughness, ao);
}
