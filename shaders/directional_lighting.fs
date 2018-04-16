#version 330 core
#pragma optionNV (unroll all)

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D accumulator;

uniform sampler2D aPosition;
uniform sampler2D aNormal;
uniform sampler2D aAlbedoSpec;

uniform samplerCube skybox;

uniform vec3 viewPos;

struct DirectionalLight {
    vec3 ambient;   // 3
    vec3 diffuse;   // 6
    vec3 specular;  // 9

    vec3 direction; // 12
    mat4 space;

    sampler2D shadowMap;
};

#define MAX_LIGHTS 1
uniform int nrLights;
uniform DirectionalLight lights[MAX_LIGHTS];

struct TextureSamples {
    vec3 diff;
    float spec;
};

float directionalShadowCalculation(float bias, sampler2D shadowMap, vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz;
    projCoords = projCoords * 0.5 + 0.5;
    float shadow;
    float currentDepth = projCoords.z;

    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    const int n = 3;
    for(int x = -n; x <= n; ++x)
    {
        for(int y = -n; y <= n; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    int m = n * 2 + 1;
    shadow /= m * m;
    // float closestDepth = texture(shadowMap, projCoors.xy).r;
    // float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    // if (currentDepth > 1.0)
    //     shadow = 0.0;

    return shadow;
}

vec3 calculateDirectionalLight(
    DirectionalLight light,
    TextureSamples samples,
    vec3 viewDir,
    vec3 normal,
    vec3 fragPos
) {
    vec3 halfwayDir = normalize(light.direction + viewDir);

    float normaldotlightdir = dot(normal, light.direction);
    float diff = max(normaldotlightdir, 0.0);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    float bias = max(0.05 * (1.0 - normaldotlightdir), 0.005);
    float shadow = 1 - directionalShadowCalculation(bias, light.shadowMap, light.space * vec4(fragPos, 1.0));

    vec3 diffuse = light.diffuse * diff * samples.diff * shadow;
    vec3 ambient = light.ambient * samples.diff;
    vec3 specular = light.specular * spec * samples.spec * shadow;

    vec3 color = diffuse + ambient + specular;

    return color;
}

void main() {
    vec3 fragPos = texture(aPosition, TexCoords).rgb;
    vec3 normal = texture(aNormal, TexCoords).rgb;
    vec4 albedoSpec = texture(aAlbedoSpec, TexCoords);

    TextureSamples samples;
    samples.diff = albedoSpec.rgb;
    samples.spec = albedoSpec.a;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-viewDir, normal);

    vec3 color;
    for (int i = 0; i < nrLights; ++i) {
        color += calculateDirectionalLight(lights[i], samples, viewDir, normal, fragPos);
    }

    FragColor = vec4(color, length(fragPos - viewPos));
}
