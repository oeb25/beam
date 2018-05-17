out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex;
uniform float scale;

vec3 sampleTex(vec2 coords) {
#if 0
    vec2 c;
    c.x = max(min(coords.x, 1.0), 0.0);
    c.y = max(min(coords.y, 1.0), 0.0);
    return scale == 2.0 ?
        pow(texture(tex, c).rgb, vec3(5.0)) :
        pow(texture(tex, c).rgb, vec3(2.0));
#else
    vec3 color = texture(tex, coords).rgb;

    if (scale != 2.0) return color;

    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 0.04)
        return color;
    else
        return vec3(0);
#endif
}

uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 c;

#if 0
    vec2 texC = TexCoords * scale;

    c += sampleTex(texC);

    vec2 range = vec2(0.02);

    int num_samples = 8;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_samples; j++) {
            float x = float(i) / float(num_samples) - 0.5;
            float y = float(j) / float(num_samples) - 0.5;

            vec2 d = vec2(x, y);

            c += sampleTex(texC + range * d) * pow(0.5 - length(d), 2);
        }
    }

#else
    vec2 tex_offset = 1.0 / textureSize(tex, 0);

    // int num_samples = 8;
    // for (int i = 0; i < num_samples * 2; i++) {
    //     for (int j = 0; j < num_samples * 2; j++) {
    //         vec2 d = vec2(i - num_samples, j - num_samples);

    //         float x = length(vec2(num_samples)) - length(d);

    //         // c += sampleTex(TexCoords + tex_offset * d) * pow((num_samples * num_samples) - length(d/float(num_samples)), 1.0/1.0);
    //         // c += sampleTex(TexCoords + tex_offset * d) * pow(num_samples - length(d), 5.0/1.0);
    //         c += sampleTex(TexCoords + tex_offset * d) * pow(x, 1.0/10.0);
    //     }
    // }

    c = sampleTex(TexCoords) * weight[0]; // current fragment's contribution
    bool horizontal = mod(scale, 2.0) == 0.0;
    if(horizontal)
    {
        for(int i = 1; i < 5; ++i)
        {
            c += sampleTex(TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            c += sampleTex(TexCoords - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        }
    }
    else
    {
        for(int i = 1; i < 5; ++i)
        {
            c += sampleTex(TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            c += sampleTex(TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        }
    }
    // FragColor = vec4(result, 1.0);

    FragColor = vec4(c, 1.0);
#endif
}
