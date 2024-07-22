#pragma once

#ifndef GLM_H
#define GLM_H
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#endif

#include <stb_image.h>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <iostream>
#include <future>
#include <execution>

enum TextureType {
	BASE = 0,
	METALLIC_ROUGHNESS = 1,
	NORMAL = 2,
	EMISSIVE = 3,
	OCCLUSION = 4
};

std::string getMimeType(fastgltf::MimeType mimeType);

class Texture {
public:
	TextureType type;
	unsigned char* pixels;
	int texWidth, texHeight, texChannels;
	std::array<float, 4> factor = { 0, 0, 0, 0 };
	uint64_t textureIndex, texCoordIndex;

	bool load(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex);

	bool load_from_loaded_texture_data(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex);

	bool load_from_file(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex, std::string parentPath);
};

class Material {
public:
	Texture baseColorTex, metalRoughTex, normalTex, emissiveTex, occlusionTex;
	bool hasBase = false, hasNormal = false, hasMR = false, hasEmissive = false, hasOcclusion = false, isAlphaModeMask = false;
};

class Drawable {
public:
	std::vector<std::uint32_t> indices;
	std::vector<glm::vec3> pos;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec4> tangents;
	std::vector<glm::vec2> texCoords;
	Material mat;
	bool hasTangents = false;
	fastgltf::TRS transformData;
};

class GLTFParser {
public:

	fastgltf::PrimitiveType renderingMode;

	std::vector<Drawable> drawables;

	void glTFError(fastgltf::Error error);

	void validateGLTF(fastgltf::Asset& asset);

	void parse(std::filesystem::path path);

	void parse_sponza(std::filesystem::path path);

private:
	std::mutex drawablesMutex;

	void processPrimitive(fastgltf::Primitive& primitive, fastgltf::Asset& asset);
};

