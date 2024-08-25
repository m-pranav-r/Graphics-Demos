#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;

layout (set = 0, binding = 1) uniform sampler2D heightMapTex;

layout (location = 0) out vec4 outColor;

void main(){
	float baseColor = texture(heightMapTex, inTexCoord).x;
	outColor = vec4(baseColor, baseColor, baseColor, 1.0);
}