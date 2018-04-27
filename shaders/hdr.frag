#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform float time;
uniform sampler2D hdrBuffer;
uniform sampler2D shadowMap;

void main() {
    const float gamma = 2.2;
    const float exposure = 0.6;
    float lookDepth = texture(hdrBuffer, vec2(0.5, 0.5)).a;
    vec4 sample = texture(hdrBuffer, TexCoords);

    float center = pow(1.0 - length(TexCoords - vec2(0.5)), 1.0 / 2.0);

    float deltaDepth = pow(sample.a - lookDepth, 1.0 / 2.0);
    vec3 color;

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

    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));
    color = color * center;

    if (length(TexCoords - vec2(0.5)) < 0.001) {
        color = vec3(1.0);
    }

    color = sample.rgb;

    bool pip = false;
    if (pip && TexCoords.x < 0.25 && TexCoords.y < 0.25) {
        color = texture(shadowMap, TexCoords * 4.0).rgb;
    }

    FragColor = vec4(color * center, 1.0);
}
