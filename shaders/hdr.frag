out vec4 FragColor;

in vec2 TexCoords;

uniform float time;
uniform sampler2D hdrBuffer;
uniform sampler2D blur[6];
// uniform sampler2D blur2;
// uniform sampler2D blur3;
// uniform sampler2D blur4;
// uniform sampler2D blur5;
// uniform sampler2D blur6;
uniform sampler2D shadowMap;

void main() {
/*
    float lookDepth = texture(hdrBuffer, vec2(0.5, 0.5)).a;

    float center = pow(1.0 - length(TexCoords - vec2(0.5)), 1.0 / 2.0);

    float deltaDepth = pow(sample.a - lookDepth, 1.0 / 2.0);

    float weights[] = float[](
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    );

    for (int i = 0; i < 9; ++i) {
        int x = i % 3;
        int y = i / 3;
        x -= 1;
        y -= 1;
        float weight = weights[i];
        vec2 tex = TexCoords + vec2(x, y) / (600.0 / deltaDepth);
        tex.x = max(min(tex.x, 1.0), 0.0);
        tex.y = max(min(tex.y, 1.0), 0.0);
        color += texture(hdrBuffer, tex).rgb * weight;
    }

    color /= 16.0;

    color = color * center;

    if (length(TexCoords - vec2(0.5)) < 0.001) {
        color = vec3(1.0);
    }
*/
    vec4 c = texture(hdrBuffer, TexCoords);
    vec3 color;

    const float gamma = 2.2;
    const float exposure = 5.0;

    color = c.rgb;

#if 1
    #define blur(n) (texture(blur[n], TexCoords).rgb * 1.0 / pow(float(n) + 1.0, 1.0/2.0))
    color += blur(0);
    color += blur(1);
    color += blur(2);
    color += blur(3);
    color += blur(4);
    color += blur(5);
#endif

    color = vec3(1.0) - exp(-color * exposure);

    // color = vec3(texture(blur[5], TexCoords));

    // color = pow(color, vec3(1.0 / gamma));

    FragColor = vec4(color, 1.0);
}
