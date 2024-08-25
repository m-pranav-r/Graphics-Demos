#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;

layout(std140, set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 viewProj;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTexCoord;


void main(){
	outNormal = inNormal;
	outTexCoord = inTexCoord;
	gl_Position = ubo.viewProj * ubo.model * vec4(inPos, 1.0);
}