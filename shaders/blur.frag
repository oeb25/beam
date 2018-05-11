#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex;
uniform float scale;

vec3 sample(vec2 coords) {
    vec2 c;
    c.x = max(min(coords.x, 1.0), 0.0);
    c.y = max(min(coords.y, 1.0), 0.0);
    return pow(texture(tex, c).rgb * 1.2, vec3(1.9));
}

void main() {
    vec3 c;

    vec2 texC = TexCoords * scale;

    c += sample(texC);

    vec2 range = vec2(0.04);

    int num_samples = 10;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_samples; j++) {
            float x = float(i) / float(num_samples) - 0.5;
            float y = float(j) / float(num_samples) - 0.5;

            vec2 d = vec2(x, y);

            c += sample(texC + range * d) * pow(0.5 - length(d), 2);
        }
    }

    FragColor = vec4(c, 1.0);
}
