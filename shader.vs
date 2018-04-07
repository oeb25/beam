#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in float aOffset;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// uniform mat4 models[100];

void main() {
	int i = int(floor(aOffset));

    float x = float(i % 100);
    float y = float(i / 100);
	FragPos = vec3(model * vec4(aPos + vec3(x, y, 0.0), 1.0));
	Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}