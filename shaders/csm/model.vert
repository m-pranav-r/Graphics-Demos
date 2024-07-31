#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in vec2 inTexCoord;

layout(std140, set = 1, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 light;
	vec3 camPos;
	vec3 lightPos;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTexCoord;
layout (location = 2) out vec4 outLightSpaceCoord;

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
	gl_Position = ubo.proj * ubo.view * pos;
	outLightSpaceCoord = biasMat * ubo.light * pos;
}