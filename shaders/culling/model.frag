#version 450

layout(location = 1) in vec2 texCoord;

layout(binding = 1) uniform sampler2D baseColorTex;

layout(location = 0) out vec4 outColor;

void main(){
	vec3 baseColor = pow(texture(baseColorTex, texCoord), vec4(2.2)).rgb;

	outColor = vec4(baseColor, 1.0);
}