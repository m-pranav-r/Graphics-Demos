#version 450

layout (location = 0) in vec3 localPos;

layout (push_constant) uniform RoughnessDetails {
	uniform float roughness;
} rd;

layout (binding = 1) uniform samplerCube cubemap;

layout (location = 0) out vec4 outColor;

const float M_PI = 3.14159265359;

float D_GGX(float NdotH, float roughness){
	float alpha = roughness * roughness;
	float alpha_squared = alpha * alpha;
	float NdotH2 = NdotH * NdotH;

	float num = alpha_squared;
	float denom = (NdotH2 * (alpha_squared - 1.0) + 1.0);
	denom = M_PI * denom * denom;

	return num / denom;
}

float RadicalInverse_Vdc(uint bits){
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

	return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N){
	return vec2(float(i) / float(N), RadicalInverse_Vdc(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness){
	float a = roughness * roughness;

	float phi = 2.0 * M_PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sintheta = sqrt(1.0 - cosTheta * cosTheta);

	vec3 H;
	H.x = cos(phi) * sintheta;
	H.y = sin(phi) * sintheta;
	H.z = cosTheta;

	vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

void main(){
	vec3 N = normalize(localPos);
	vec3 R = N;
	vec3 V = N;

	const uint noSamples = 1024u;
	float totalWeight = 0.0;
	vec3 prefilteredColor = vec3(0.0);
	for(uint i = 0u; i < noSamples; i++){
		vec2 Xi = Hammersley(i, noSamples);
		vec3 H = ImportanceSampleGGX(Xi, N, rd.roughness);
		vec3 L = normalize(2.0 * dot(V, H) * H - V);
		float NdotH = max(dot(N, H), 0.0);
		float D = D_GGX(NdotH, rd.roughness);
		float pdf = (D * NdotH / (4.0 * max(dot(H, V), 0.0))) + 0.0001;

		float resolution = 1024;
		float saTexel = 4.0 * M_PI / (6.0 * resolution * resolution);
		float saSample = 1.0 / (float(noSamples) * pdf + 0.0001);

		float mipLevel = rd.roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

		float NdotL = max(dot(N, L), 0.0);
		if(NdotL > 0.0){
			prefilteredColor += textureLod(cubemap, L, mipLevel).rgb * NdotL;
			totalWeight += NdotL;
		}
	}
	prefilteredColor = prefilteredColor / totalWeight;

	outColor = vec4(prefilteredColor, 1.0);
}