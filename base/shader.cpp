#include "shader.h"

void ShaderHelper::createShaderModule(std::vector<uint32_t> code) {
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();// *sizeof(uint32_t);
	createInfo.pCode = code.data();

	if (code.size() == 0) {
		throw std::runtime_error("sda");
	}

	VkResult shaderModuleCreateResult = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
	if (shaderModuleCreateResult != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}
}

void ShaderHelper::compileShaderFromFile(const std::string& srcName, const std::string& dstName) {
	int returnCode = system(std::format("C:\\VulkanSDK\\1.3.283.0\\Bin\\glslc.exe {} -o {}", srcName, dstName).c_str());
	if(returnCode != 0){
		throw std::runtime_error("failed to dynamic compile shaders!\n");
	}
}

void ShaderHelper::compileShaderToSPIRVAndCreateShaderModule()
{

	if (glslBuffer.size() == 0) {
		throw std::runtime_error("ShaderHelper::compileShaderToSPIRV() called without file read first\n");
	}

	shaderc::Compiler compiler;
	shaderc::CompileOptions compileOptions;

	if (shaderDefinitions.size() != 0) {
		for (auto iter = shaderDefinitions.begin(); iter != shaderDefinitions.end(); iter++) {
			compileOptions.AddMacroDefinition(iter->data());
		}
	}

	shaderc_shader_kind shaderKind = shaderc_shader_kind::shaderc_callable_shader;

	switch (type)
	{
	case VERT:
		shaderKind = shaderc_shader_kind::shaderc_glsl_vertex_shader;
		break;
	case FRAG:
		shaderKind = shaderc_shader_kind::shaderc_glsl_fragment_shader;
		break;
	default:
		break;
	}

	//compile preprocessed glsl to spirv
	auto module = compiler.CompileGlslToSpv(glslBuffer.data(), glslBuffer.size(), shaderKind, shaderName.c_str());
	auto compileResult = module.GetCompilationStatus();
	if (compileResult != shaderc_compilation_status_success) {
		std::cout << "failed to compile glsl to spv! error: \n" << module.GetErrorMessage() << std::endl;
		throw std::runtime_error("error from shader.cpp\n");
	}

	spirvBuffer = { module.cbegin(), module.cend() };

	createShaderModule(spirvBuffer);
}

void ShaderHelper::readCompiledSPIRVAndCreateShaderModule(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}
	size_t fileSize = (size_t)file.tellg();
	spirvBuffer.resize(fileSize);

	file.seekg(0);
	file.read(reinterpret_cast<char*>(spirvBuffer.data()), fileSize);
	file.close();

	createShaderModule(spirvBuffer);
}

void ShaderHelper::init(std::string name, Type type, VkDevice device)
{
	this->shaderName = name;
	this->type = type;
	this->device = device;
}

void ShaderHelper::addShaderDefinitionTerm(const char* term) {
	shaderDefinitions.push_back(term);
}

std::vector<char> ShaderHelper::readFileHead(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	std::vector<char> headBuffer; char i = 'i';

	while (i != '\n') {
		i = file.get();
		headBuffer.push_back(i);
	}

	i = 0;
	return headBuffer;
}

void ShaderHelper::readFileGLSL(const std::string& fileName) {
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}
	size_t fileSize = (size_t)file.tellg();
	glslBuffer.resize(fileSize);

	file.seekg(0);
	file.read(glslBuffer.data(), fileSize);
	file.close();
}