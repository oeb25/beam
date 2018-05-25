in vec4 FragPos;
out vec4 FragColor;

uniform vec3 lightPos;
uniform float farPlane;

void main() {
    float lightDistance = length(lightPos - FragPos.xyz);

    lightDistance /= farPlane;

    gl_FragDepth = lightDistance;
}
