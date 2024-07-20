#version 450

layout (location = 0) in vec3 pos;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) out vec3 fragPos;

void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
	fragPos = pos;
}