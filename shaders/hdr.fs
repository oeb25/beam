#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D hdrBuffer;

void main() {
    const float gamma = 2.2;

    vec3 color = texture(hdrBuffer, TexCoords).rgb;
    color = color / (color + vec3(1));
    color = pow(color, vec3(1.0 / gamma));
    FragColor = vec4(color, 1.0);
}
