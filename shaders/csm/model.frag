#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec4 inLightSpaceCoord;

layout (set = 1, binding = 1) uniform sampler2D baseColorTex;
layout (set = 1, binding = 2) uniform sampler2D metallicRoughness;
layout (set = 1, binding = 3) uniform sampler2D normals;

layout (set = 0, binding = 0) uniform sampler2D shadowTex;

layout (location = 0) out vec4 outColor;

float ShadowCalculation(vec4 projCoords){
	/*
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

	float closestDepth = texture(shadowTex, projCoords.xy).r;
	float currentDepth = projCoords.z;

	return currentDepth >= closestDepth ? 1.0 : 0.0;
	*/
	vec4 shadowCoords = projCoords / projCoords.w;
	//shadowCoords = shadowCoords * 0.5 + 0.5;
	float closestDepth = texture(shadowTex, shadowCoords.st).r;
	float currentDepth = shadowCoords.z;

	float shadow = currentDepth - 0.005 > closestDepth ? 0.15 : 1.0;

	return shadow;
}

void main(){
	vec4 baseColor = pow(texture(baseColorTex, inTexCoord), vec4(2.2));
	//outColor = vec4(ShadowCalculation(inLightSpaceCoord).xxx, 1.0);
	outColor = vec4(baseColor.xyz * ShadowCalculation(inLightSpaceCoord), 1.0);
}