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

struct PointLight {
    vec3 ambient;    // 3
    vec3 diffuse;    // 6
    vec3 specular;   // 9

    vec3 position;   // 12
    vec3 lastPosition;   // 12

    float constant;  // 13
    float linear;    // 14
    float quadratic; // 15

    samplerCube shadowMap;
    float farPlane;
};

#define MAX_LIGHTS 4
uniform int nrLights;
uniform PointLight lights[MAX_LIGHTS];

struct TextureSamples {
    vec3 diff;
    float spec;
};

float pointShadowCalculation(float bias, float farPlane, samplerCube shadowMap, vec3 lightPos, vec3 fragPos) {
    vec3 fragToLight = fragPos - lightPos;
    float closestDepth = texture(shadowMap, fragToLight).r;
    closestDepth *= farPlane;
    float currentDepth = length(fragToLight);

    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

    return shadow;
}

vec3 calculatePointLight(
    PointLight light,
    TextureSamples samples,
    vec3 viewDir,
    vec3 normal,
    vec3 fragPos
) {
    vec3 lightDir = normalize(light.position - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float d = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + d * (light.linear + d * light.quadratic));

    float diff = max(dot(normal, lightDir), 0.0);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    float bias = 0.05;
    float shadow = 1 - pointShadowCalculation(bias, light.farPlane, light.shadowMap, light.lastPosition, fragPos);

    vec3 ambient = light.ambient * samples.diff;
    vec3 diffuse = light.diffuse * diff * samples.diff * shadow;
    vec3 specular = light.specular * spec * samples.spec * shadow;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

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

    vec3 color = texture(accumulator, TexCoords).rgb;
    for (int i = 0; i < nrLights; ++i) {
        color += calculatePointLight(lights[i], samples, viewDir, normal, fragPos);
    }

    FragColor = vec4(color, length(fragPos - viewPos));
}
