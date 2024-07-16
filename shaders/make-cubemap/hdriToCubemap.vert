#version 450
#extension GL_EXT_multiview: enable

layout(location = 0) in vec3 aPos;

layout(set = 0, binding = 0) uniform CubemapBufferObject{
	mat4 model;
	mat4 view[6];
	mat4 proj;
} cbo;

layout(location = 0) out vec3 localPos;

void main(){
	localPos = aPos;
	gl_Position = cbo.proj * cbo.view[gl_ViewIndex] * vec4(localPos, 1.0);
}