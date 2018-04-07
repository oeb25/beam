#version 330 core

out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;

uniform sampler2D diffuseTex;
uniform sampler2D specularTex;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 lightDir = normalize(lightPos - FragPos);
	vec3 halfwayDir = normalize(lightDir + viewDir);

	float diff = max(dot(Normal, lightDir), 0.0);

	float ambient = 0.1;

	float spec = pow(max(dot(Normal, halfwayDir), 0.0), 32.0);

    FragColor = texture(diffuseTex, TexCoords) * (diff + ambient) + texture(specularTex, TexCoords) * spec;
}