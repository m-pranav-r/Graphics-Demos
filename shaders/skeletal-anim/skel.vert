#version 450

layout (location = 0) in vec3 pos;
layout (location = 1) in vec4 joint;
layout (location = 2) in vec4 weights;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

struct JointData{
	mat4 matrix;
};
layout(set = 0, binding = 1, std140) readonly buffer JointBuffer{
	JointData joints[];
};
void main(){
	mat4 skinMat = 
		weights.x * joints[int(joint.x)].matrix +
		weights.y * joints[int(joint.y)].matrix +
		weights.z * joints[int(joint.z)].matrix +
		weights.w * joints[int(joint.w)].matrix;
	gl_Position = ubo.proj * ubo.view * ubo.model * skinMat * vec4(pos, 1.0);
}