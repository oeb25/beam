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

#define def_mat(typ, name) \
    uniform bool use_mat_##name; \
    uniform typ mat_##name; \
    uniform sampler2D tex_##name

def_mat(vec3, albedo);
def_mat(vec3, normal);
def_mat(float, metallic);
def_mat(float, roughness);
def_mat(float, ao);
def_mat(float, opacity);

uniform vec3 viewPos;

void main() {
    #define tex(name, fn) (use_mat_##name ? mat_##name : fn(texture(tex_##name, fs_in.TexCoords)))

    vec3 norm = tex(normal, vec3);
    norm = normalize(norm * 2.0 - 1.0);
    norm = normalize(fs_in.TBN * norm);

    vec3 albedo = tex(albedo, vec3);

    float m = tex(metallic, float);
    float r = tex(roughness, float);
    float a = tex(ao, float);
    float o = tex(opacity, float);

    vec4 mrao = vec4(m, r, a, o);

    if (mrao.a < 0.1) {
        discard;
    }

    aPosition = fs_in.FragPos;
    aNormal.rgb = norm;
    aAlbedo.rgb = albedo;
    aMrao.rgba = mrao;
}
