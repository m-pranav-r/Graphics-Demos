#version 450

layout(location = 0) in vec3 aPos;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec3 camPos;
} ubo;

layout(location = 0) out vec3 texCoords;

void main(){
	texCoords = aPos;
	//texCoords.xy *= -1.0;
	mat4 statView = mat4(mat3(ubo.view));
	gl_Position = ubo.proj * statView * vec4(aPos, 1.0);
}