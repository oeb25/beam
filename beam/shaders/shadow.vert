layout (location = 0) in vec3 aPos;
layout (location = 5) in mat4 aModel;

uniform mat4 lightSpace;

void main() {
    vec4 pos = aModel * vec4(aPos, 1.0);
    pos.z = 1;
    gl_Position = lightSpace * pos;
}
