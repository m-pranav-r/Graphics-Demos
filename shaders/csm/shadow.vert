#version 450

layout(location = 0) in vec3 aPos;

layout(push_constant) uniform lightSpaceMatrix{
	mat4 model;
	mat4 lightSpaceMatrix;
} pushConstant;

void main(){
	gl_Position = (pushConstant.lightSpaceMatrix * pushConstant.model) * vec4(aPos, 1.0);
}