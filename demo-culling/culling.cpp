#include "init.h"

#include "parser.h"
#include "camera.h"
#include "device.h"
#include "memory.h"
#include "command.h"
#include "shader.h"

#ifndef GLM_H
#define GLM_H
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#endif

#include <chrono>
#include <cstdlib>
#include <set>
#include <optional>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <array>
#include <unordered_map>


#ifdef _DEBUG
bool isDebugEnv = true;
#else
bool isDebugEnv = false;
#endif

const uint32_t INSTANCE_COUNT = 64;

struct Vertex {
	glm::vec3 pos;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

struct InstanceData {
	glm::vec3 pos;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 1;
		bindingDescription.stride = sizeof(InstanceData);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		return bindingDescription;
	}

	static VkVertexInputAttributeDescription getAttributeDescriptions() {
		VkVertexInputAttributeDescription attributeDescription{};

		attributeDescription.binding = 1;
		attributeDescription.location = 2;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = offsetof(InstanceData, pos);

		return attributeDescription;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

std::vector<float> cubeVertices = {
	// back face
	-1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	// front face
	-1.0f, -1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,
	// left face
	-1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	// right face
	 1.0f,  1.0f,  1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	 // bottom face
	 -1.0f, -1.0f, -1.0f,
	  1.0f, -1.0f, -1.0f,
	  1.0f, -1.0f,  1.0f,
	  1.0f, -1.0f,  1.0f,
	 -1.0f, -1.0f,  1.0f,
	 -1.0f, -1.0f, -1.0f,
	 // top face
	 -1.0f,  1.0f, -1.0f,
	  1.0f,  1.0f , 1.0f,
	  1.0f,  1.0f, -1.0f,
	  1.0f,  1.0f,  1.0f,
	 -1.0f,  1.0f, -1.0f,
	 -1.0f,  1.0f,  1.0f,
};

const std::vector<const char*> validationlayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
	VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
	VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
	VK_KHR_MULTIVIEW_EXTENSION_NAME,
#ifdef _DEBUG_NVIDIA
	VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME,
	VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME
#endif
};

const int MAX_FRAMES_IN_FLIGHT = 2;

struct DrawableHandle {
	VkImage baseColorImage;
	VkImageView baseColorImageView;
	VkSampler baseColorSampler;
	VkDeviceMemory baseColorMemory;
	std::vector<Vertex> vertices;
	VkBuffer vertexBuffer, indexBuffer;
	VkDeviceMemory vertexBufferMemory, indexBufferMemory;
	size_t indices;

	//bounds data
	VkBuffer boundingBoxVertexBuffer;
	VkDeviceMemory boundingBoxVertexBufferMemory;
};

Camera camera;

class CullingDemo {
public:
	CullingDemo(uint32_t width, uint32_t height, std::vector<Drawable>& drawables) : WIDTH(width), HEIGHT(height), drawables(drawables) {};

	void run() {
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	uint32_t WIDTH, HEIGHT;
	std::vector<Drawable> drawables;

	VkInstance instance;

	Init vulkanInit;

	DeviceHelper deviceHelper;

	MemoryHelper memHelper;			std::mutex memMutex;

	CommandHelper commHelper;		std::mutex commMutex;

	std::chrono::steady_clock::time_point lastTime;

	const char* hdriPath;

	VkSurfaceKHR surface;
	GLFWwindow* window;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue transferQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	std::vector<VkDescriptorSet> descriptorSets;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	VkCommandPool transferPool;
	VkDescriptorPool descriptorPool;
	uint32_t mipLevels;

	std::vector<DrawableHandle> drawableHandles;
	std::mutex drawableHandlesMutex;

	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	VkDescriptorSetLayout fullDSL;

	//bounding render data
	VkPipelineLayout boundingBoxLayout;
	VkPipeline boundingBoxPipeline;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkBuffer offscreenUniformBuffer;
	VkDeviceMemory offscreenUniformBufferMemory;
	void* offscreenUniformBufferMapped;

	VkBuffer instanceBuffer;
	VkDeviceMemory instanceBufferMemory;

	std::array<InstanceData, INSTANCE_COUNT * INSTANCE_COUNT> instanceGenData;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore >imageAvailableSemaphores;
	std::vector<VkSemaphore >renderFinishedSemaphores;
	std::vector<VkFence >inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;

	//glm::vec3 sponzaScaleMatrix = glm::vec3(0.000800000038);

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<CullingDemo*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {

		vulkanInit.isDebug = isDebugEnv;
		vulkanInit.width = WIDTH;
		vulkanInit.height = HEIGHT;
		vulkanInit.title = "Instancing - No UI";
		vulkanInit.addLayer("VK_LAYER_KHRONOS_validation");
		vulkanInit.addInstanceExtension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		vulkanInit.setFramebufferResizeFunc(framebufferResizeCallback);
		vulkanInit.setCursorCallback([](GLFWwindow* window, double x, double y) { camera.processGLFWMouseEvent(window, x, y); });
		vulkanInit.setKeyboardCallback([](GLFWwindow* window, int key, int scancode, int action, int mods) {camera.processGLFWKeyboardEvent(window, key, scancode, action, mods); });
		vulkanInit.init();

		//set camera stuff here
		camera.velocity = glm::vec3(0.f);
		camera.position = glm::vec3(2.f);
		camera.pitch = 0;
		camera.yaw = 0;

		instance = vulkanInit.getInstance();
		window = vulkanInit.getWindow();
		surface = vulkanInit.getSurface();

		//pickPhysicalDevice();

		deviceHelper.instance = instance;
		deviceHelper.surface = surface;
		deviceHelper.addLayer("VK_LAYER_KHRONOS_validation");
		deviceHelper.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		deviceHelper.initDevices();
		deviceHelper.createSwapchain(swapChain, window, swapChainImages, swapChainImageFormat, swapChainExtent);
		device = deviceHelper.getDevice();
		physicalDevice = deviceHelper.getPhysicalDevice();

		deviceHelper.getQueues(graphicsQueue, presentQueue, transferQueue);

		memHelper.init(
			deviceHelper.getQueueFamilyIndices(),
			deviceHelper.getDevice(),
			deviceHelper.getPhysicalDevice()
		);

		commHelper.init(
			deviceHelper.getQueueFamilyIndices(),
			deviceHelper.getDevice(),
			deviceHelper.getPhysicalDevice()
		);

		createImageViews();

		createDescriptorSetLayouts();
		createGraphicsPipelines();

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			QUEUE_TYPE_GRAPHICS,
			commandPool
		);

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			QUEUE_TYPE_TRANSFER,
			transferPool
		);

		//createColorResources();

		memHelper.createImage(
			swapChainExtent.width,
			swapChainExtent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			swapChainImageFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			colorImage,
			colorImageMemory
		);

		colorImageView = createImageView(colorImage, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		memHelper.createImage(
			swapChainExtent.width,
			swapChainExtent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			deviceHelper.findDepthFormat(),
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depthImage,
			depthImageMemory
		);
		depthImageView = createImageView(depthImage, deviceHelper.findDepthFormat(), VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(commandPool);

		memHelper.transitionImageLayout(commandBuffer, depthImage, deviceHelper.findDepthFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

		commHelper.endSingleTimeCommands(commandBuffer, commandPool, graphicsQueue);

		processDrawables();

		createInstanceBuffers();

		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
		std::cout << "done!\n\n\n";
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			VkResult swapChainImageViewCreateResult = vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]);
			if (swapChainImageViewCreateResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createDescriptorSetLayouts() {

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding baseColorLayoutBinding{};
		baseColorLayoutBinding.binding = 1;
		baseColorLayoutBinding.descriptorCount = 1;
		baseColorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		baseColorLayoutBinding.pImmutableSamplers = nullptr;
		baseColorLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		std::vector<VkDescriptorSetLayoutBinding> bindingsAll = {
			uboLayoutBinding,
			baseColorLayoutBinding,
		};
		layoutInfo.bindingCount = static_cast<uint32_t>(bindingsAll.size());
		layoutInfo.pBindings = bindingsAll.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &fullDSL) != VK_SUCCESS) {
			throw std::runtime_error("failed to create full descriptor set layout!");
		}
	}

	void createGraphicsPipelines() {

		//normal pipeline
		ShaderHelper vertShader;
		vertShader.init("vert", Type::VERT, device);
		vertShader.readCompiledSPIRVAndCreateShaderModule("../shaders/culling/model.vert.spv");

		ShaderHelper fragShader;
		fragShader.init("frag", Type::FRAG, device);
		fragShader.readCompiledSPIRVAndCreateShaderModule("../shaders/culling/model.frag.spv");

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShader.shaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShader.shaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto vertexBindingDescription = Vertex::getBindingDescription();
		auto vertexAttributeDescription = Vertex::getAttributeDescriptions();

		auto instanceBindingDescription = InstanceData::getBindingDescription();
		auto instanceAttributeDescription = InstanceData::getAttributeDescriptions();

		VkVertexInputBindingDescription bindingDescriptions[] = { vertexBindingDescription, instanceBindingDescription };
		VkVertexInputAttributeDescription attributeDescriptions[] = { vertexAttributeDescription[0], vertexAttributeDescription[1], instanceAttributeDescription};
		

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 2;
		vertexInputInfo.vertexAttributeDescriptionCount = 3;
		vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;

		pipelineLayoutInfo.pSetLayouts = &fullDSL;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainImageFormat,
			.depthAttachmentFormat = deviceHelper.findDepthFormat()
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
		pipelineInfo.pStages = shaderStages;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShader.shaderModule, nullptr);
		vkDestroyShaderModule(device, fragShader.shaderModule, nullptr);

		/*

		//bounding box pipeline
		ShaderHelper boundingVertShader, boundingFragShader;
		boundingVertShader.init("bounding vert", Type::VERT, device);
		boundingVertShader.readCompiledSPIRVAndCreateShaderModule("../shaders/culling/boundingbox.vert.spv");
		boundingFragShader.init("bounding frag", Type::FRAG, device);
		boundingFragShader.readCompiledSPIRVAndCreateShaderModule("../shaders/culling/boundingbox.frag.spv");

		vertShaderStageInfo.module = boundingVertShader.shaderModule;
		fragShaderStageInfo.module = boundingFragShader.shaderModule;

		VkPipelineShaderStageCreateInfo boundingShaderStages[] = {vertShaderStageInfo , fragShaderStageInfo};
		
		VkVertexInputBindingDescription boundingVertexBindingDesc{};
		boundingVertexBindingDesc.binding = 0;
		boundingVertexBindingDesc.stride = sizeof(float) * 3;
		boundingVertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription boundingVertexAttribDesc{};
		boundingVertexAttribDesc.binding = 0;
		boundingVertexAttribDesc.location = 0;
		boundingVertexAttribDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
		boundingVertexAttribDesc.offset = 0;

		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &boundingVertexBindingDesc;
		vertexInputInfo.pVertexAttributeDescriptions = &boundingVertexAttribDesc;

		rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
		rasterizer.cullMode = VK_CULL_MODE_NONE;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &boundingBoxLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create boundind box pipeline layout!");
		}

		pipelineInfo.layout = boundingBoxLayout;
		pipelineInfo.pStages = boundingShaderStages;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &boundingBoxPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create bounding box pipeline!");
		}

		vkDestroyShaderModule(device, boundingVertShader.shaderModule, nullptr);
		vkDestroyShaderModule(device, boundingFragShader.shaderModule, nullptr);
		*/
	}

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

		if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
		if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
		if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
		if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
		if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
		if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;

		return VK_SAMPLE_COUNT_1_BIT;
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;

		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	void ProcessDrawable(Drawable drawable) {

		DrawableHandle currDrawableHandle;

		std::vector<Texture> textures = {
			drawable.mat.baseColorTex,
		};

		VkCommandPool tempPool;

		commMutex.lock();

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			QUEUE_TYPE_GRAPHICS,
			tempPool
		);

		commMutex.unlock();

		for (auto& texture : textures) {
			mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texture.texWidth, texture.texHeight)))) + 1;

			VkDeviceSize imageSize = texture.texWidth * texture.texHeight * 4;

			if (!texture.pixels) {
				throw std::runtime_error("failed to load texture image from memory!");
			}

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			memMutex.lock();

			memHelper.createBuffer(
				imageSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer,
				stagingBufferMemory,
				QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
			);

			void* data;
			if (vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data) != VK_SUCCESS) {
				throw std::runtime_error("failed to map texture memory!");
			}
			memcpy(data, texture.pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(device, stagingBufferMemory);

			stbi_image_free(texture.pixels);

			VkImage* textureImage = nullptr;
			VkDeviceMemory* textureImageMemory = nullptr;
			VkImageView* textureImageView = nullptr;
			VkSampler* textureSampler = nullptr;
			VkFormat textureImageFormat;

			textureImage = &currDrawableHandle.baseColorImage;
			textureImageMemory = &currDrawableHandle.baseColorMemory;
			textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
			textureImageView = &currDrawableHandle.baseColorImageView;
			textureSampler = &currDrawableHandle.baseColorSampler;


			memHelper.createImage(
				texture.texWidth,
				texture.texHeight,
				1,
				VK_SAMPLE_COUNT_1_BIT,
				textureImageFormat,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				*textureImage,
				*textureImageMemory
			);

			VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

			memHelper.transitionImageLayout(
				commandBuffer,
				*textureImage,
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1
			);

			memHelper.copyBufferToImage(
				commandBuffer,
				stagingBuffer,
				*textureImage,
				static_cast<uint32_t>(texture.texWidth),
				static_cast<uint32_t>(texture.texHeight)
			);

			commMutex.lock();

			commHelper.endSingleTimeCommands(commandBuffer, tempPool, graphicsQueue);

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);

			commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

			memHelper.transitionImageLayout(
				commandBuffer,
				*textureImage,
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				1
			);

			memMutex.unlock();

			commHelper.endSingleTimeCommands(commandBuffer, tempPool, graphicsQueue);

			commMutex.unlock();

			//generateMipmaps(*textureImage, textureImageFormat, texture.texWidth, texture.texHeight, mipLevels);

			//create tex image views
			*textureImageView = createImageView(*textureImage, textureImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

			//create tex samplers
			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;

			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(physicalDevice, &properties);

			samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.minLod = 0.0f;
			samplerInfo.maxLod = static_cast<float>(1);
			samplerInfo.mipLodBias = 0.0f;

			if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture sampler!");
			}
		}

		glm::vec3 min{ 999.0f }, max{ -999.0f };

		for (int j = 0; j < drawable.pos.size(); j++) {
			Vertex vertex{};

			vertex.pos = drawable.pos[j];
			vertex.texCoord = drawable.texCoords[j];

			//bounding box generation
			min = glm::vec3(
				std::min(min[0], vertex.pos[0]),
				std::min(min[1], vertex.pos[1]),
				std::min(min[2], vertex.pos[2])
			);
			
			max = glm::vec3(
				std::max(max[0], vertex.pos[0]),
				std::max(max[1], vertex.pos[1]),
				std::max(max[2], vertex.pos[2])
			);

			currDrawableHandle.vertices.push_back(vertex);
		}

		drawable.pos.clear();
		drawable.pos.shrink_to_fit();
		drawable.normals.clear();
		drawable.normals.shrink_to_fit();
		drawable.tangents.clear();
		drawable.tangents.shrink_to_fit();
		drawable.texCoords.clear();
		drawable.texCoords.shrink_to_fit();

		std::array<float, 3 * 6 * 6> boundingBoxVertices = {
			// back face
			min[0], min[1], min[2],
			max[0], max[1], min[2],
			max[0], min[1], min[2],
			max[0], max[1], min[2],
			min[0], min[1], min[2],
			min[0], max[1], min[2],
			// front face
			min[0], min[1], max[2],
			max[0], min[1], max[2],
			max[0], max[1], max[2],
			max[0], max[1], max[2],
			min[0], max[1], max[2],
			min[0], min[1], max[2],
			// left face
			min[0], max[1], max[2],
			min[0], max[1], min[2],
			min[0], min[1], min[2],
			min[0], min[1], max[2],
			min[0], min[1], min[2],
			min[0], max[1], max[2],
			// right face
			max[0], max[1], max[2],
			max[0], min[1], min[2],
			max[0], max[1], min[2],
			max[0], min[1], min[2],
			max[0], max[1], max[2],
			max[0], min[1], max[2],
			 // bottom face
			min[0], min[1], min[2],
			max[0], min[1], min[2],
			max[0], min[1], max[2],
			max[0], min[1], max[2],
			min[0], min[1], max[2],
			min[0], min[1], min[2],
			// top face
			min[0], max[2], min[2],
			max[0], max[2] ,max[2],
			max[0], max[2], min[2],
			max[0], max[2], max[2],
			min[0], max[2], min[2],
			min[0], max[2], max[2],
		};

		//create vertex buffer
		VkDeviceSize vertexBufferSize = sizeof(currDrawableHandle.vertices[0]) * currDrawableHandle.vertices.size();

		VkBuffer stagingVertexBuffer;
		VkDeviceMemory stagingVertexBufferMemory;

		memMutex.lock();

		memHelper.createBuffer(
			vertexBufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingVertexBuffer,
			stagingVertexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);

		void* vertexData;
		vkMapMemory(device, stagingVertexBufferMemory, 0, vertexBufferSize, 0, &vertexData);
		memcpy(vertexData, currDrawableHandle.vertices.data(), (size_t)vertexBufferSize);
		vkUnmapMemory(device, stagingVertexBufferMemory);

		memHelper.createBuffer(
			vertexBufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			currDrawableHandle.vertexBuffer,
			currDrawableHandle.vertexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);
		VkCommandBuffer commandBuffer;
		commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

		memHelper.copyBuffer(commandBuffer, stagingVertexBuffer, currDrawableHandle.vertexBuffer, vertexBufferSize);

		commMutex.lock();

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, graphicsQueue);	//kooky

		vkDestroyBuffer(device, stagingVertexBuffer, nullptr);
		vkFreeMemory(device, stagingVertexBufferMemory, nullptr);

		currDrawableHandle.vertices.clear();
		currDrawableHandle.vertices.shrink_to_fit();

		//create index buffer
		VkDeviceSize indexBufferSize = sizeof(drawable.indices[0]) * drawable.indices.size();
		currDrawableHandle.indices = drawable.indices.size();
		VkBuffer stagingIndexBuffer;
		VkDeviceMemory stagingIndexBufferMemory;
		memHelper.createBuffer(
			indexBufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingIndexBuffer,
			stagingIndexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);

		void* indexData;
		vkMapMemory(device, stagingIndexBufferMemory, 0, indexBufferSize, 0, &indexData);
		memcpy(indexData, drawable.indices.data(), (size_t)indexBufferSize);
		vkUnmapMemory(device, stagingIndexBufferMemory);

		memHelper.createBuffer(
			indexBufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			currDrawableHandle.indexBuffer,
			currDrawableHandle.indexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);
		commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

		memHelper.copyBuffer(commandBuffer, stagingIndexBuffer, currDrawableHandle.indexBuffer, indexBufferSize);

		memMutex.unlock();

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, graphicsQueue);	//kooky 2

		commMutex.unlock();

		vkDestroyBuffer(device, stagingIndexBuffer, nullptr);
		vkFreeMemory(device, stagingIndexBufferMemory, nullptr);

		//create bounding box buffer
		{
			VkDeviceSize boundingBoxBufferSize = sizeof(boundingBoxVertices[0]) * boundingBoxVertices.size();

			VkBuffer stagingVertexBuffer;
			VkDeviceMemory stagingVertexBufferMemory;

			memMutex.lock();

			memHelper.createBuffer(
				boundingBoxBufferSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingVertexBuffer,
				stagingVertexBufferMemory,
				QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
			);

			commMutex.lock();

			commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

			void* boundingData;
			vkMapMemory(device, stagingVertexBufferMemory, 0, boundingBoxBufferSize, 0, &boundingData);
			memcpy(boundingData, boundingBoxVertices.data(), (size_t)boundingBoxBufferSize);
			vkUnmapMemory(device, stagingVertexBufferMemory);

			memHelper.createBuffer(
				boundingBoxBufferSize,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				currDrawableHandle.boundingBoxVertexBuffer,
				currDrawableHandle.boundingBoxVertexBufferMemory,
				QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
			);
			VkCommandBuffer commandBuffer;
			commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

			memHelper.copyBuffer(commandBuffer, stagingVertexBuffer, currDrawableHandle.boundingBoxVertexBuffer, boundingBoxBufferSize);

			memMutex.unlock();

			commHelper.endSingleTimeCommands(commandBuffer, tempPool, graphicsQueue);	//kooky

			commMutex.unlock();

			vkDestroyBuffer(device, stagingVertexBuffer, nullptr);
			vkFreeMemory(device, stagingVertexBufferMemory, nullptr);
		}

		drawable.indices.clear();
		drawable.indices.shrink_to_fit();

		drawableHandlesMutex.lock();

		drawableHandles.push_back(currDrawableHandle);

		drawableHandlesMutex.unlock();

		vkDestroyCommandPool(device, tempPool, nullptr);
	}

	void processDrawables() {
		auto makeBuffersTask = std::async(std::launch::async,
			[&]() {
				std::for_each(std::execution::par,
				drawables.begin(),
				drawables.end(),
				[&](Drawable drawable) {
						ProcessDrawable(drawable);
					}
		);
			}
		);

		makeBuffersTask.wait();

		vkDestroyCommandPool(device, transferPool, nullptr);

		drawables.clear();
		drawables.shrink_to_fit();

		std::cerr << "\nAll drawables processed.\n";
	}

	void createInstanceBuffers() {
		const uint32_t totalInstances = INSTANCE_COUNT * INSTANCE_COUNT;
		VkDeviceSize bufferSize = sizeof(InstanceData) * totalInstances;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingDeviceMemory;

		memHelper.createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingDeviceMemory,
			QUEUE_TYPE_GRAPHICS
		);

		void* stagingPointer;
		vkMapMemory(device, stagingDeviceMemory, 0, bufferSize, 0, &stagingPointer);

		for (int i = 0; i < totalInstances / INSTANCE_COUNT; i++) {
			for (int j = 0; j < totalInstances / INSTANCE_COUNT; j++) {
				int idx = i + j * INSTANCE_COUNT;
				instanceGenData[idx].pos = glm::vec3( .2f * i, 0.0f, .2f * j );
				//maybe add scaling data later
			}
		}

		memcpy(stagingPointer, instanceGenData.data(), bufferSize);

		vkUnmapMemory(device, stagingDeviceMemory);

		memHelper.createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			instanceBuffer,
			instanceBufferMemory,
			QUEUE_TYPE_GRAPHICS
		);

		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(commandPool);

		memHelper.copyBuffer(
			commandBuffer,
			stagingBuffer,
			instanceBuffer,
			bufferSize
		);

		commHelper.endSingleTimeCommands(commandBuffer, commandPool, graphicsQueue);

		vkFreeMemory(device, stagingDeviceMemory, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			memHelper.createBuffer(
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i],
				uniformBuffersMemory[i],
				QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
			);
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void createDescriptorPool() {
		VkDescriptorPoolSize uniformPoolInfo = {
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 150
		};

		VkDescriptorPoolSize texturePoolInfo = {
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 320
		};

		std::array<VkDescriptorPoolSize, 2> poolSizes = { uniformPoolInfo, texturePoolInfo };

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.poolSizeCount = 2;
		poolInfo.maxSets = 300;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets() {
		{
			std::vector<VkDescriptorSetLayout> layouts(drawableHandles.size() * MAX_FRAMES_IN_FLIGHT, fullDSL);

			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
			allocInfo.pSetLayouts = layouts.data();
			descriptorSets.resize(layouts.size());

			VkResult allocResult = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
			if (allocResult != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}
		}

		for (size_t j = 0; j < drawableHandles.size(); j++) {
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				VkDescriptorBufferInfo uniformBufferInfo{};
				uniformBufferInfo.buffer = uniformBuffers[i];
				uniformBufferInfo.offset = 0;
				uniformBufferInfo.range = sizeof(UniformBufferObject);

				VkDescriptorImageInfo baseColorImageInfo{};
				baseColorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				baseColorImageInfo.imageView = drawableHandles[j].baseColorImageView;
				baseColorImageInfo.sampler = drawableHandles[j].baseColorSampler;

				std::array<VkWriteDescriptorSet, 2> descWrites{};

				descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[0].dstSet = descriptorSets[(2 * j) + i];
				descWrites[0].dstBinding = 0;
				descWrites[0].dstArrayElement = 0;
				descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descWrites[0].descriptorCount = 1;
				descWrites[0].pBufferInfo = &uniformBufferInfo;

				descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[1].dstSet = descriptorSets[(2 * j) + i];
				descWrites[1].dstBinding = 1;
				descWrites[1].dstArrayElement = 0;
				descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descWrites[1].descriptorCount = 1;
				descWrites[1].pImageInfo = &baseColorImageInfo;

				vkUpdateDescriptorSets(device, 2, descWrites.data(), 0, nullptr);
			}
		}
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command buffers!");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { {0.4f, 0.5f, 0.6f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderingAttachmentInfoKHR colorAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = swapChainImageViews[imageIndex],
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = clearValues[0]
		};

		VkRenderingAttachmentInfoKHR depthAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = depthImageView,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.clearValue = clearValues[1]
		};

		VkRenderingInfoKHR renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, swapChainExtent.width, swapChainExtent.height},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment,
			.pDepthAttachment = &depthAttachment,
		};

		//color image transition
		VkImageMemoryBarrier colorImageStartTransitionBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapChainImages[imageIndex],
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&colorImageStartTransitionBarrier
		);

		//depth image transition
		VkImageMemoryBarrier depthImageStartTransitionBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = depthImage,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&depthImageStartTransitionBarrier
		);

		vkCmdBeginRendering(commandBuffer, &renderingInfo);


		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		//render cubemap		
		/*
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapPipeline);

		VkBuffer cubemapBuffers[] = { offscreenVertexBuffer };

		vkCmdBindVertexBuffers(commandBuffer, 0, 1, cubemapBuffers, offsets);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapPipelineLayout, 0, 1, &cubemapDescriptorSet, 0, nullptr);
		vkCmdDraw(commandBuffer, 36, 1, 0, 0);
		*/

		VkDeviceSize offsets[] = { 0 };

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		//render model
		for (int i = 0; i < drawableHandles.size(); i++) {
			VkBuffer vertexBuffers[] = { drawableHandles[i].vertexBuffer };
			VkBuffer instanceBuffers[] = { instanceBuffer };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindVertexBuffers(commandBuffer, 1, 1, instanceBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, drawableHandles[i].indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[(2 * i) + currentFrame], 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(drawableHandles[i].indices), INSTANCE_COUNT * INSTANCE_COUNT, 0, 0, 0);

			/*
			vertexBuffers[0] = drawableHandles[i].boundingBoxVertexBuffer;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, boundingBoxPipeline);
			vkCmdDraw(commandBuffer, 3 * 6 * 6, 1, 0, 0);
			*/
		}


		vkCmdEndRendering(commandBuffer);

		//transitionImageLayout();

		//transition images to present_src
		VkImageMemoryBarrier colorImageEndTransitionBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapChainImages[imageIndex],
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&colorImageEndTransitionBarrier
		);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create sync objects for a frame!");
			}
		}
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
		
		auto latestTime = std::chrono::high_resolution_clock::now();
		updateUniformBuffer(currentFrame, std::chrono::duration<float, std::chrono::seconds::period>(latestTime - lastTime).count());
		auto lastTime = latestTime;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage, float deltaTime) {
		camera.update(deltaTime);

		UniformBufferObject ubo{};
		ubo.model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0));
		ubo.view = camera.getViewMatrix();
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.01f, 100.0f);
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		//createSwapChain();
		//add routine here to recreate device swapchain
		createImageViews();
		//createColorResources();
		//createDepthResources();
		//createFramebuffers();
	}

	void cleanupSwapChain() {
		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto swapChainImageView : swapChainImageViews) {
			vkDestroyImageView(device, swapChainImageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void destroyDrawableHandleData(DrawableHandle drawableHandle) {

		vkDestroySampler(device, drawableHandle.baseColorSampler, nullptr);
		vkDestroyImageView(device, drawableHandle.baseColorImageView, nullptr);
		vkDestroyImage(device, drawableHandle.baseColorImage, nullptr);
		vkFreeMemory(device, drawableHandle.baseColorMemory, nullptr);

		vkDestroyBuffer(device, drawableHandle.indexBuffer, nullptr);
		vkFreeMemory(device, drawableHandle.indexBufferMemory, nullptr);

		vkDestroyBuffer(device, drawableHandle.vertexBuffer, nullptr);
		vkFreeMemory(device, drawableHandle.vertexBufferMemory, nullptr);

		vkDestroyBuffer(device, drawableHandle.boundingBoxVertexBuffer, nullptr);
		vkFreeMemory(device, drawableHandle.boundingBoxVertexBufferMemory, nullptr);
	}

	void cleanup() {
		cleanupSwapChain();

		auto deleteTask = std::async(std::launch::async,
			[&]() {
				std::for_each(std::execution::par,
				drawableHandles.begin(),
				drawableHandles.end(),
				[&](DrawableHandle drawableHandle) {
						destroyDrawableHandleData(drawableHandle);
					}
		);
			}
		);

		vkDestroyBuffer(device, instanceBuffer, nullptr);
		vkFreeMemory(device, instanceBufferMemory, nullptr);

		deleteTask.wait();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorSetLayout(device, fullDSL, nullptr);

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkDestroyPipeline(device, boundingBoxPipeline, nullptr);
		vkDestroyPipelineLayout(device, boundingBoxLayout, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		vulkanInit.destroy();

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main() {
	try {
		GLTFParser sponzaParser;
		sponzaParser.parse("../models/WaterBottle.glb");
		CullingDemo app(1280, 720, sponzaParser.drawables);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}