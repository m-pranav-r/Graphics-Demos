#pragma once

#include "vulkan/vulkan.h"

#include "shaderc/shaderc.hpp"

#include <iostream>
#include <format>
#include <fstream>
#include <vector>

enum Type {
	VERT = 0,
	FRAG = 1,
	COMP = 2
};

class ShaderHelper {
public:

	void addShaderDefinitionTerm(const char* term);

	void readFileGLSL(const std::string& fileName);

	std::vector<char> readFileHead(const std::string& fileName);

	void compileShaderFromFile(const std::string& srcName, const std::string& dstName);

	void compileShaderToSPIRVAndCreateShaderModule();

	void readCompiledSPIRVAndCreateShaderModule(const std::string& fileName);

	void init(std::string name, Type type, VkDevice device);

	void createShaderModule(std::vector<uint32_t> code);

	VkShaderModule shaderModule;

private:

	std::vector<char> glslBuffer;

	std::vector<uint32_t> spirvBuffer;

	std::vector<std::string> shaderDefinitions;

	std::string shaderName;

	Type type;

	VkDevice device;
};