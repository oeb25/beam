layout (location = 0) in vec3 aPos;
layout (location = 5) in mat4 aModel;

uniform mat4 model;

void main() {
    mat4 model_ = aModel * model;
    gl_Position = model_ * vec4(aPos, 1.0);
}
