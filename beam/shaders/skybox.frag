out vec4 FragColor;

in vec3 TexCoords;

uniform float skyboxIntensity;
uniform samplerCube skybox;

void main() {
    vec3 color = textureLod(skybox, TexCoords, 0.0).rgb * skyboxIntensity;

    FragColor = vec4(color, 1.0);
}
