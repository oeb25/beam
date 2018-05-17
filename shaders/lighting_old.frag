#pragma optionNV (unroll all)

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D aPosition;
uniform sampler2D aNormal;
uniform sampler2D aAlbedoSpec;

uniform samplerCube skybox;
uniform sampler2D shadowMap;
uniform samplerCube pointShadowMap;
uniform float farPlane;

uniform vec3 viewPos;

struct DirectionalLight {
    vec3 ambient;   // 3
    vec3 diffuse;   // 6
    vec3 specular;  // 9

    vec3 direction; // 12
    mat4 space;
};

struct PointLight {
    vec3 ambient;    // 3
    vec3 diffuse;    // 6
    vec3 specular;   // 9

    vec3 position;   // 12

    float constant;  // 13
    float linear;    // 14
    float quadratic; // 15
};

struct SpotLight {
    vec3 ambient;      // 3
    vec3 diffuse;      // 6
    vec3 specular;     // 9

    vec3 position;     // 12
    vec3 direction;    // 15

    float cutOff;      // 16
    float outerCutOff; // 17

    float constant;    // 18
    float linear;      // 19
    float quadratic;   // 20
};

#define MAX_DIRECTIONAL_LIGHTS 10
#define MAX_POINT_LIGHTS       10
#define MAX_SPOT_LIGHTS        10

uniform int numDirectionalLights;
uniform DirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
uniform int numPointLights;
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int numSpotLights;
uniform SpotLight spotLights[MAX_SPOT_LIGHTS];

struct TextureSamples {
    vec3 diff;
    float spec;
    vec3 refl;
    vec3 reflectSkybox;
};

float directionalShadowCalculation(float bias, vec4 fragPosLightSpace) {
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
    float shadow = 1 - directionalShadowCalculation(bias, light.space * vec4(fragPos, 1.0));

    vec3 diffuse = light.diffuse * diff * samples.diff * shadow;
    vec3 ambient = light.ambient * samples.diff;
    vec3 specular = light.specular * spec * samples.spec * shadow;

    vec3 color = diffuse + ambient + specular;

        // color = vec3(shadow);

    return color;
}

float pointShadowCalculation(float bias, vec3 lightPos, vec3 fragPos) {
    vec3 fragToLight = fragPos - lightPos;
    float closestDepth = texture(pointShadowMap, fragToLight).r;
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
    float shadow = 1 - pointShadowCalculation(bias, light.position, fragPos);

    vec3 ambient = light.ambient * samples.diff;
    vec3 diffuse = light.diffuse * diff * samples.diff * shadow;
    vec3 specular = light.specular * spec * samples.spec * shadow;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    vec3 color = diffuse + ambient + specular;

    return color;
}

vec3 calculateSpotLight(
    SpotLight light,
    TextureSamples samples,
    vec3 viewDir,
    vec3 normal,
    vec3 fragPos
) {
    vec3 lightDir = normalize(light.position - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);

    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    float d = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + d * (light.linear + d * light.quadratic));

    float diff = max(dot(normal, lightDir), 0.0);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    vec3 ambient = light.ambient * samples.diff;
    vec3 diffuse = light.diffuse * diff * samples.diff;
    vec3 specular = light.specular * spec * samples.spec;

    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;

    vec3 color = diffuse + ambient + specular;

    return color;
}

void main() {
    const float gamma = 2.2;

    vec3 fragPos = texture(aPosition, TexCoords).rgb;
    vec3 normal = texture(aNormal, TexCoords).rgb;
    vec4 albedoSpec = texture(aAlbedoSpec, TexCoords);

    TextureSamples samples;
    samples.diff = albedoSpec.rgb;
    samples.spec = albedoSpec.a;
    // samples.refl = texture(tex_reflection1, fs_in.TexCoords).rgb;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-viewDir, normal);
    // vec3 reflectDir = refract(-viewDir, normalize(normal), 1.0 / 1.52);

    vec3 color;

    color += calculateDirectionalLight(directionalLights[0], samples, viewDir, normal, fragPos);
    // for (int i = 0; i < numDirectionalLights; ++i) {
    //     color += calculateDirectionalLight(directionalLights[i], samples, viewDir, normal, fragPos);
    // }
    for (int i = 0; i < numPointLights; ++i) {
        color += calculatePointLight(pointLights[i], samples, viewDir, normal, fragPos);
    }
    for (int i = 0; i < numSpotLights; ++i) {
        color += calculateSpotLight(spotLights[i], samples, viewDir, normal, fragPos);
    }

    // color += texture(skybox, reflectDir).rgb * 0.01;

    FragColor = vec4(color, length(fragPos - viewPos));
}
