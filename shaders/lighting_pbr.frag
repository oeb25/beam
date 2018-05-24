#pragma optionNV (unroll all)

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D aPosition;
uniform sampler2D aNormal;
uniform sampler2D aAlbedo;
uniform sampler2D aMetallicRoughnessAo;

uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

uniform vec3 viewPos;

// There can only be one shadow map for point lights
uniform samplerCube pointShadowMap;
uniform sampler2D directionalShadowMap;

struct PointLight {
    vec3 ambient;    // 3
    vec3 diffuse;    // 6
    vec3 specular;   // 9

    vec3 position;   // 12
    vec3 lastPosition;   // 12

    float constant;  // 13
    float linear;    // 14
    float quadratic; // 15

    bool useShadowMap;
    float farPlane;
};

struct DirectionalLight {
    vec3 ambient;   // 3
    vec3 diffuse;   // 6
    vec3 specular;  // 9

    vec3 direction; // 12
    mat4 space;
};

#define MAX_DIR_LIGHTS 1
uniform int nrDirLights;
uniform DirectionalLight directionalLights[MAX_DIR_LIGHTS];

#define MAX_POINT_LIGHTS 10
uniform int nrPointLights;
uniform PointLight pointLights[MAX_POINT_LIGHTS];

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

    return shadow;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

#define PI 3.14159265359

float DistributonGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

void main() {
    vec3 fragPos = texture(aPosition, TexCoords).rgb;
    vec3 N = normalize(texture(aNormal, TexCoords).rgb);
    vec3 metallicRoughnessAo = texture(aMetallicRoughnessAo, TexCoords).rgb;
    vec3 albedo = texture(aAlbedo, TexCoords).rgb;
    float metallic = metallicRoughnessAo.r;
    float roughness = metallicRoughnessAo.g;
    float ao = metallicRoughnessAo.b;

    // albedo = vec3(1.00, 0.26, 0.27);
    // metallic = 0.9;
    // roughness = 0.0;

    vec3 V = normalize(viewPos - fragPos);
    vec3 R = reflect(-V, N);

    #define saturate(n) (clamp(n, 0.0, 1.0))

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo;
    vec3 color;
    for (int i = 0; i < nrPointLights; ++i) {
        PointLight light = pointLights[i];
        vec3 L = normalize(light.position - fragPos);
        vec3 H = normalize(V + L);

        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light.diffuse * 100.0 * attenuation;

        float NDF = DistributonGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / max(denominator, 0.001);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;

        kD *= 1.0 - metallic;

        float shadow = 1.0;
        // calculate shadow only if the shadow map is enabled
        if (light.useShadowMap) {
            vec3 fragToLight = fragPos - light.lastPosition;
            float closestDepth = texture(pointShadowMap, fragToLight).r;
            closestDepth *= light.farPlane;
            float currentDepth = length(fragToLight);

            float bias = 0.05;

            shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
            shadow = 1 - shadow;
        }

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    }
    for (int i = 0; i < nrDirLights; ++i) {
        DirectionalLight light = directionalLights[i];
        vec3 L = normalize(light.direction);
        vec3 H = normalize(V + L);

        vec3 radiance = light.diffuse * 20.0;

        vec3 F = fresnelSchlick(max(dot(N, V), 0.0), F0);

        float NDF = DistributonGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular = numerator / max(denominator, 0.001);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;

        kD *= 1.0 - metallic;

        float NdotL = max(dot(N, L), 0.0);
        float bias = max(0.05 * (1.0 - NdotL), 0.0002);
        float shadow = 1.0 - directionalShadowCalculation(0.0, directionalShadowMap, light.space * vec4(fragPos, 1.0));

        Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    }

    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;

    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    color = ambient + Lo;

    FragColor = vec4(color, length(fragPos - viewPos));
}
