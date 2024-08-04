#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec4 inWorldPos;
layout (location = 3) in vec4 inViewPos;

layout (set = 1, binding = 1) uniform sampler2D baseColorTex;
layout (set = 1, binding = 2) uniform sampler2D metallicRoughness;
layout (set = 1, binding = 3) uniform sampler2D normals;

layout (set = 0, binding = 0) uniform sampler2D shadowTex[4];

layout(std140, set = 0, binding = 1) uniform ShadowUniformBufferObject{
	mat4 light[4];
	vec4 far;
} subo;

layout(constant_id = 0) const int usePCF = 0;
layout(constant_id = 1) const int debugColors = 0;

layout (location = 0) out vec4 outColor;

const float ambient = 0.05;

float PCFShadowCalc(vec4 projCoords, int shadowIndex){
	vec4 shadowCoords = projCoords / projCoords.w;

	ivec2 texDim = textureSize(shadowTex[shadowIndex], 0);
	vec2 texelSize = 1.0 / texDim;

	float shadow = 0.0;
	for(int x = -1; x <= 1; x++){
		for(int y = -1; y <= 1; y++){
			float pcfDepth = texture(shadowTex[shadowIndex], shadowCoords.st + texelSize * vec2(x, y)).r;
			shadow += shadowCoords.z - 0.005 > pcfDepth ? 1.0 : ambient;
		}
	}
	return shadow / 9;
}

float ShadowCalculation(vec4 projCoords, int shadowIndex){
	vec4 shadowCoords = projCoords / projCoords.w;

	float closestDepth = texture(shadowTex[shadowIndex], shadowCoords.st).r;
	float currentDepth = shadowCoords.z;

	float shadow = currentDepth - 0.005 > closestDepth ? 1.0 : ambient;

	return shadow;
}

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
);

void main(){
	int shadowIndex = 0;
	float depth = inViewPos.z;
	if(depth >= subo.far.x) shadowIndex = 1;
	if(depth >= subo.far.y) shadowIndex = 2;
	if(depth >= subo.far.z) shadowIndex = 3;
	vec4 lightSpaceCoord = biasMat * subo.light[shadowIndex] * inWorldPos;

	if(debugColors == 1){
		vec3 color = 
				shadowIndex == 0 ? vec3(1.0, 0.0, 0.0) :	
				shadowIndex == 1 ? vec3(0.0, 1.0, 0.0) :	
				shadowIndex == 2 ? vec3(0.0, 0.0, 1.0) :	vec3(1.0);
		outColor = vec4(color, 1.0);
	} else {
		vec4 baseColor = pow(texture(baseColorTex, inTexCoord), vec4(2.2));
		outColor = vec4(baseColor.xyz * (1 - (usePCF == 1 ? PCFShadowCalc(lightSpaceCoord, shadowIndex) : ShadowCalculation(lightSpaceCoord, shadowIndex))), 1.0);
	}
}