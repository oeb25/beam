// out vec4 FragColor;

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

// Material
uniform bool useMaterial;
uniform vec3 mat_albedo;
uniform vec3 mat_metallicRoughnessAo;

uniform bool useNormalMap;
uniform vec3 viewPos;

vec3 xxx() {
    vec3 tangentNormal = texture(tex_normal, fs_in.TexCoords).xyz * 2.0 - 1.0;
    // tangentNormal = vec3(0.0, 0.0, 0.0);

    vec3 Q1  = dFdx(fs_in.FragPos);
    vec3 Q2  = dFdy(fs_in.FragPos);
    vec2 st1 = dFdx(fs_in.TexCoords);
    vec2 st2 = dFdy(fs_in.TexCoords);

    vec3 N   = normalize(fs_in.Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

void main() {
    vec3 norm;
    if (useNormalMap || true) {
        norm = texture(tex_normal, fs_in.TexCoords).rgb;
        norm = normalize(norm * 2.0 - 1.0);
        norm = normalize(fs_in.TBN * norm);
    } else {
        norm = normalize(fs_in.Normal);
    }

    vec3 albedo;
    float metallic;
    float roughness;
    float ao;

    aPosition = fs_in.FragPos;
    aNormal.rgb = norm;
    if (useMaterial && false) {
        albedo = mat_albedo;
        metallic = mat_metallicRoughnessAo.r;
        roughness = mat_metallicRoughnessAo.g;
        ao = mat_metallicRoughnessAo.b;
    } else {
        albedo = texture(tex_albedo, fs_in.TexCoords).rgb;
        roughness = texture(tex_roughness, fs_in.TexCoords).r;
        metallic = texture(tex_metallic, fs_in.TexCoords).r;
        ao = texture(tex_ao, fs_in.TexCoords).r;
    }

    aAlbedo.rgb = albedo;
    aMetallicRoughnessAo.rgb = vec3(metallic, roughness, ao);
}
