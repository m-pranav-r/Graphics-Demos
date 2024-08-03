#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in vec2 inTexCoord;

layout(std140, set = 1, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 viewProj;

	mat4 light[4];
	float far[4];

	vec3 camPos;
	vec3 lightPos;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTexCoord;
layout (location = 2) out vec4 outLightSpaceCoord;
layout (location = 3) flat out int shadowIndex;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
);

void main(){
	outNormal = inNormal;
	outTexCoord = inTexCoord;
	vec4 pos = ubo.model * vec4(inPos, 1.0);
	gl_Position = ubo.viewProj * pos;
	float depth = gl_Position.z;
	shadowIndex = 0;
	if(depth > ubo.far[0]) shadowIndex = 1;
	if(depth > ubo.far[1]) shadowIndex = 2;
	if(depth > ubo.far[2]) shadowIndex = 3;
	outLightSpaceCoord = biasMat * ubo.light[shadowIndex] * pos;
}