out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D display;

void main() {
    vec3 c;

    c.rgb = texture(display, TexCoords).rgb;

    FragColor = vec4(c, 1.0);
}
