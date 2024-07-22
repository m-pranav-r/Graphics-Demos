#version 450

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 texCoord;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout (location = 2) in vec3 Instpos;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec2 fragTexCoord;

void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos - Instpos, 1.0);
	fragPos = pos;
	fragTexCoord = texCoord;
}