#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec4 inLightSpaceCoord;
layout (location = 3) flat in int inShadowIndex;

layout (set = 1, binding = 1) uniform sampler2D baseColorTex;
layout (set = 1, binding = 2) uniform sampler2D metallicRoughness;
layout (set = 1, binding = 3) uniform sampler2D normals;

layout (set = 0, binding = 0) uniform sampler2D shadowTex[4];

layout(constant_id = 0) const int usePCF = 0;

layout (location = 0) out vec4 outColor;

const float ambient = 0.05;

float PCFShadowCalc(vec4 projCoords){
	vec4 shadowCoords = projCoords / projCoords.w;

	ivec2 texDim = textureSize(shadowTex[inShadowIndex], 0);
	vec2 texelSize = 1.0 / texDim;

	float shadow = 0.0;
	for(int x = -1; x <= 1; x++){
		for(int y = -1; y <= 1; y++){
			float pcfDepth = texture(shadowTex[inShadowIndex], shadowCoords.st + texelSize * vec2(x, y)).r;
			shadow += shadowCoords.z - 0.005 > pcfDepth ? 1.0 : ambient;
		}
	}
	return shadow / 9;
}

float ShadowCalculation(vec4 projCoords){
	vec4 shadowCoords = projCoords / projCoords.w;

	float closestDepth = texture(shadowTex[inShadowIndex], shadowCoords.st).r;
	float currentDepth = shadowCoords.z;

	float shadow = currentDepth - 0.005 > closestDepth ? 1.0 : ambient;

	return shadow;
}

void main(){
	vec4 baseColor = pow(texture(baseColorTex, inTexCoord), vec4(2.2));
	outColor = vec4(baseColor.xyz * (1 - (usePCF == 1 ? PCFShadowCalc(inLightSpaceCoord) : ShadowCalculation(inLightSpaceCoord))), 1.0);
	/*
	vec3 color = 
			inShadowIndex == 0 ? vec3(1.0, 0.0, 0.0) :	
			inShadowIndex == 1 ? vec3(0.0, 1.0, 0.0) :	
			inShadowIndex == 2 ? vec3(0.0, 0.0, 1.0) :	vec3(1.0);
	outColor = vec4(color, 1.0);
	*/
}