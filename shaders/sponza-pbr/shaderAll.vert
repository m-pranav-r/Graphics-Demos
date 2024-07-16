#version 450

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 tangent;
layout (location = 3) in vec2 texCoord;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec3 camPos;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragWorldPos;
layout(location = 4) out vec3 camPos;
layout(location = 5) out mat3 TBN;

void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
	fragPos = pos;
	fragNormal = normal;
	fragTexCoord = texCoord;
	fragWorldPos = vec4(ubo.model * vec4(pos, 1.0)).xyz;
	camPos = ubo.camPos;

	//TBN calculations
	vec3 T = normalize(vec3(ubo.model * vec4(tangent.xyz, 0.0)));
	vec3 N = normalize(vec3(ubo.model * vec4(normal, 0.0)));
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T) * tangent.w;
	TBN = mat3(T, B, N);
}