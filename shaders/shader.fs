#version 410 core

out vec4 FragColor;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} fs_in;

struct DirectionalLight {
    vec3 ambient;   // 3
    vec3 diffuse;   // 6
    vec3 specular;  // 9

    vec3 direction; // 12
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
};

uniform sampler2D tex_diffuse1;
uniform sampler2D tex_specular1;
uniform sampler2D tex_reflection1;
uniform sampler2D tex_normal1;
uniform bool useNormalMap;
uniform vec3 viewPos;
uniform samplerCube skybox;

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
    vec3 spec;
    vec3 refl;
    vec3 reflectSkybox;
};

vec3 calculateDirectionalLight(
    DirectionalLight light,
    TextureSamples samples,
    vec3 viewDir,
    vec3 normal,
    vec3 fragPos
) {
    vec3 halfwayDir = normalize(light.direction + viewDir);

    float diff = max(dot(normal, light.direction), 0.0);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    vec3 diffuse = light.diffuse * diff * samples.diff;
    vec3 ambient = light.ambient * samples.diff;
    vec3 specular = light.specular * spec * samples.spec;

    vec3 color = diffuse + ambient + specular;

    return color;
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

    vec3 ambient = light.ambient * samples.diff;
    vec3 diffuse = light.diffuse * diff * samples.diff;
    vec3 specular = light.specular * spec * samples.spec;

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
    // float attenuation = 1.0 / (light.constant + d * (light.linear + d * light.quadratic));

    float diff = max(dot(normal, lightDir), 0.0);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    vec3 ambient = light.ambient * samples.diff;
    vec3 diffuse = light.diffuse * diff * samples.diff;
    vec3 specular = light.specular * spec * samples.spec;

    ambient *= intensity;
    diffuse *= intensity;
    specular *= intensity;

    vec3 color = diffuse + ambient + specular;

    return color;
}

void main() {
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);

    vec3 norm;
    // if (useNormalMap) {
        norm = texture(tex_normal1, fs_in.TexCoords).rgb;
        norm = normalize(norm * 2.0 - 1.0);
        norm = normalize(fs_in.TBN * norm);
    // } else {
    //     norm = fs_in.Normal;
    // }

    vec3 reflectDir = reflect(-viewDir, normalize(norm));
    // reflectDir = refract(-viewDir, normalize(norm), 1.0 / 1.33);

    TextureSamples samples;
    samples.diff = texture(tex_diffuse1, fs_in.TexCoords).rgb;
    samples.spec = texture(tex_specular1, fs_in.TexCoords).rgb;
    samples.refl = texture(tex_reflection1, fs_in.TexCoords).rgb;
    // samples.reflectSkybox = texture(skybox, reflectDir).rgb;

    vec3 fragPos = fs_in.FragPos;

    vec3 color;
#if 0
    for (int i = 0; i < numDirectionalLights; i++) {
        color +=
            calculateDirectionalLight(directionalLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < numPointLights; i++) {
        color +=
            calculatePointLight(pointLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < numSpotLights; i++) {
        color +=
            calculateSpotLight(spotLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
#endif
#if 0
    for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
        color +=
            calculateDirectionalLight(directionalLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
        color +=
            calculatePointLight(pointLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < MAX_SPOT_LIGHTS; i++) {
        color +=
            calculateSpotLight(spotLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
#endif
#if 0
    for (int i = 0; i < 1; i++) {
        color +=
            calculateDirectionalLight(directionalLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < 1; i++) {
        color +=
            calculatePointLight(pointLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
    for (int i = 0; i < 1; i++) {
        color +=
            calculateSpotLight(spotLights[i], samples, viewDir, norm, reflectDir, fragPos);
    }
#endif
#if 1
    // color += calculateDirectionalLight(directionalLights[0], samples, viewDir, norm, fragPos);
    color += calculatePointLight(pointLights[0], samples, viewDir, norm, fragPos);
    color += calculatePointLight(pointLights[1], samples, viewDir, norm, fragPos);
    // color += calculateSpotLight(spotLights[0], samples, viewDir, norm, fragPos);
#endif

    color += length(color) * samples.reflectSkybox * samples.refl * 0.9;

    // color = reflectDir;

    // color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
