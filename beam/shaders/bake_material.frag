out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex_metallic;
uniform sampler2D tex_roughness;
uniform sampler2D tex_ao;
uniform sampler2D tex_opacity;

void main() {
    float metallic = texture(tex_metallic, TexCoords).r;
    float roughness = texture(tex_roughness, TexCoords).r;
    float ao = texture(tex_ao, TexCoords).r;
    float opacity = texture(tex_opacity, TexCoords).r;

    FragColor = vec4(metallic, roughness, ao, opacity);
}
