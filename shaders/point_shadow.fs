#version 330 core

in vec4 FragPos;
out vec4 FragColor;

uniform vec3 lightPos;
uniform float farPlane;

void main() {
    float lightDistance = length(lightPos - FragPos.xyz);

    lightDistance /= farPlane;

    gl_FragDepth = lightDistance;
    // FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}
