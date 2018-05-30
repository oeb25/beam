out vec4 FragColor;

in vec2 TexCoords;

uniform float time;
uniform sampler2D hdrBuffer;
uniform sampler2D blur[6];
uniform sampler2D shadowMap;

void main() {
    vec4 c = texture(hdrBuffer, TexCoords);
    vec3 color;

    const float gamma = 2.2;
    const float exposure = 0.1;

    color = c.rgb;

#if 1
    #define blur(n) (0.5 * (texture(blur[n], TexCoords).rgb * 1.0 / pow(float(n) + 1.0, 1.0/2.0)))
    // color += blur(0);
    // color += blur(1);
    // color += blur(2);
    // color += blur(3);
    // color += blur(4);
    // color += blur(5);
#endif

    color = color / (vec3(1.0) + color);

    color = pow(color, vec3(1.0 / gamma));

    FragColor = vec4(color, 1.0);
}
