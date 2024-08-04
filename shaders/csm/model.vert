#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in vec2 inTexCoord;

layout(std140, set = 1, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 viewProj;

	mat4 light[4];
	vec4 far;

	vec3 camPos;
	vec3 lightPos;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTexCoord;
layout (location = 2) out vec4 outWorldPos;
layout (location = 3) out vec4 outViewPos;


void main(){
	outNormal = inNormal;
	outTexCoord = inTexCoord;
	outWorldPos = ubo.model * vec4(inPos, 1.0);
	outViewPos = ubo.viewProj * outWorldPos;
	gl_Position = outViewPos;
}