#version 450

layout (location = 0) in vec3 pos;
layout (location = 1) in vec4 joint;
layout (location = 2) in vec4 weights;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 joints[2];
	vec3 camPos;
} ubo;

void main(){
	mat4 skinMat = 
		weights.x * ubo.joints[int(joint.x)] +
		weights.y * ubo.joints[int(joint.y)] +
		weights.z * ubo.joints[int(joint.z)] +
		weights.w * ubo.joints[int(joint.w)];
	gl_Position = ubo.proj * ubo.view * ubo.model * skinMat * vec4(pos, 1.0);
}