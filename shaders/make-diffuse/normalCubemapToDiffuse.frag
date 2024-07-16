#version 450

layout(location = 0) in vec3 localPos;

layout(binding = 1) uniform samplerCube cubemap;

layout(location = 0) out vec4 outColor;

const float M_PI = 3.14159265359;

void main(){
	vec3 normal = normalize(localPos);

	vec3 irradiance = vec3(0.0);

	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = normalize(cross(up, normal));
	up = normalize(cross(normal, right));

	float sampleDelta = 0.025;
	float noSamples = 0.0;

	for(float phi = 0.0; phi < 2.0 * M_PI; phi += sampleDelta){
		for(float theta = 0.0; theta < 0.5 * M_PI; theta += sampleDelta){
			vec3 tangentSample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
			vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;

			irradiance += texture(cubemap, sampleVec).rgb * cos(theta) * sin(theta);
			noSamples++;
		}
	}

	irradiance = M_PI * irradiance * (1.0 / float(noSamples));

	outColor = vec4(irradiance, 1.0);
}