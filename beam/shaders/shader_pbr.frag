layout (location = 0) out vec3 aPosition;
layout (location = 1) out vec4 aNormal;
layout (location = 2) out vec4 aAlbedo;
layout (location = 3) out vec4 aMrao;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} fs_in;

#define use_mrao 0

uniform bool use_mat_albedo;
uniform vec3 mat_albedo;
uniform bool use_mat_normal;
uniform vec3 mat_normal;
uniform bool use_mat_metallic;
uniform float mat_metallic;
uniform bool use_mat_roughness;
uniform float mat_roughness;
uniform bool use_mat_ao;
uniform float mat_ao;
uniform bool use_mat_opacity;
uniform float mat_opacity;

uniform sampler2D tex_albedo;
uniform sampler2D tex_normal;
uniform sampler2D tex_metallic;
uniform sampler2D tex_roughness;
uniform sampler2D tex_ao;
uniform sampler2D tex_opacity;

uniform vec3 viewPos;

void main() {
    #define tex(fn, name) (use_mat_##name ? mat_##name : fn(texture(tex_##name, fs_in.TexCoords)))

    vec3 norm = tex(vec3, normal);
    norm = normalize(norm * 2.0 - 1.0);
    norm = normalize(fs_in.TBN * norm);

    vec3 albedo = tex(vec3, albedo);

    float m = tex(float, metallic);
    float r = tex(float, roughness);
    float a = tex(float, ao);
    float o = tex(float, opacity);

    vec4 mrao = vec4(m, r, a, o);

    if (mrao.a < 0.1) {
        discard;
    }

    aPosition = fs_in.FragPos;
    aNormal.rgb = norm;
    aAlbedo.rgb = albedo;
    aMrao.rgba = mrao;
}
