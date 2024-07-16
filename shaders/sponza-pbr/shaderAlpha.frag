#version 450

//#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 worldPos;
layout(location = 4) in vec3 camPos;
layout(location = 5) in mat3 TBN;

layout(binding = 1) uniform sampler2D baseColorTex;

layout(location = 0) out vec4 outColor;

void main(){
	vec4 colorSample = texture(baseColorTex, texCoord);
	if(colorSample.a < 0.1) discard;
	outColor = vec4(colorSample);
}