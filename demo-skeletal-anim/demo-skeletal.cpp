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

struct Vertex {
	glm::vec3 pos;
	glm::vec4 joints;
	glm::vec4 weights;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, joints);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, weights);

		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 joints[2];
	glm::vec3 camPos;
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

static std::vector<char> readFile(const std::string& fileName) {
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);


	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
}

Camera camera;

bool playAnim = false;
double scalingFactor = 0.0001;

class SkeletalAnimTest {
public:
	SkeletalAnimTest(uint32_t width, uint32_t height, Drawable model) : WIDTH(width), HEIGHT(height), model(model) {};

	void run() {
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	uint32_t WIDTH, HEIGHT;
	Drawable model;

	VkInstance instance;
	Init vulkanInit;
	DeviceHelper deviceHelper;
	MemoryHelper memHelper;			std::mutex memMutex;
	CommandHelper commHelper;		std::mutex commMutex;

	VkSurfaceKHR surface;
	GLFWwindow* window;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	vk::Device device;

	struct {
		VkQueue graphics;
		VkQueue present;
		VkQueue transfer;
	} queues;

	struct {
		VkSwapchainKHR swapchain;
		std::vector<VkImage> images;
		VkFormat imageFormat;
		VkExtent2D extent;
		std::vector<vk::ImageView> imageViews;
	} swapchain;

	struct {
		VkRenderPass renderPass;
		VkPipelineLayout pipelineLayout;
		std::vector<VkDescriptorSet> descriptorSets;
		VkPipeline graphicsPipeline;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout dsl;
	} renderState;

	struct {
		VkCommandPool command;
		VkCommandPool transfer;
	} pools;

	struct {
		VkImage image;
		VkDeviceMemory imageMemory;
		VkImageView imageView;
	} colorRender, depthRender;

	struct {
		std::vector<Vertex> vertices;
		VkBuffer vertexBuffer, indexBuffer;
		VkDeviceMemory vertexBufferMemory, indexBufferMemory;
		size_t indices;
	} drawData;

	struct {
		std::vector<VkBuffer> buffers;
		std::vector<VkDeviceMemory> buffersMemory;
		std::vector<void*> buffersMapped;
	} uniform;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore >imageAvailableSemaphores;
	std::vector<VkSemaphore >renderFinishedSemaphores;
	std::vector<VkFence >inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;

	std::chrono::steady_clock::time_point lastTime;
	float timeElapsed = 0.0f;

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<SkeletalAnimTest*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {

		vulkanInit.isDebug = isDebugEnv;
		vulkanInit.width = WIDTH;
		vulkanInit.height = HEIGHT;
		vulkanInit.title = "Skeletal Animation Demo";
		vulkanInit.addLayer("VK_LAYER_KHRONOS_validation");
		vulkanInit.addInstanceExtension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		vulkanInit.setFramebufferResizeFunc(framebufferResizeCallback);
		vulkanInit.setCursorCallback(
			[](GLFWwindow* window, double x, double y) { 
				camera.processGLFWMouseEvent(window, x, y); 
			}
		);
		vulkanInit.setKeyboardCallback(
			[](GLFWwindow* window, int key, int scancode, int action, int mods) {
				camera.processGLFWKeyboardEvent(window, key, scancode, action, mods); 
				if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) playAnim = !playAnim;
				if (action == GLFW_PRESS && key == GLFW_KEY_UP) scalingFactor *= 2;
				if (action == GLFW_PRESS && key == GLFW_KEY_DOWN) scalingFactor /= 2;
			}
		);
		vulkanInit.init();

		camera.velocity = glm::vec3(0.f);
		camera.position = glm::vec3(0.f, 1.f, 4.f);
		camera.pitch = 0;
		camera.yaw = 0;
		camera.scalingFactor = 2;

		instance = vulkanInit.getInstance();
		window = vulkanInit.getWindow();
		surface = vulkanInit.getSurface();

		deviceHelper.instance = instance;
		deviceHelper.surface = surface;
		deviceHelper.addLayer("VK_LAYER_KHRONOS_validation");
		deviceHelper.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		deviceHelper.initDevices();
		deviceHelper.createSwapchain(swapchain.swapchain, window, swapchain.images, swapchain.imageFormat, swapchain.extent);
		device = deviceHelper.getDevice();
		physicalDevice = deviceHelper.getPhysicalDevice();

		deviceHelper.getQueues(queues.graphics, queues.present, queues.transfer);

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
			pools.command
		);

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			QUEUE_TYPE_TRANSFER,
			pools.transfer
		);

		memHelper.createImage(
			swapchain.extent.width,
			swapchain.extent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			swapchain.imageFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			colorRender.image,
			colorRender.imageMemory
		);

		colorRender.imageView = createImageView(colorRender.image, swapchain.imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		memHelper.createImage(
			swapchain.extent.width,
			swapchain.extent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			deviceHelper.findDepthFormat(),
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depthRender.image,
			depthRender.imageMemory
		);
		depthRender.imageView = createImageView(depthRender.image, deviceHelper.findDepthFormat(), VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(pools.command);

		memHelper.transitionImageLayout(commandBuffer, depthRender.image, deviceHelper.findDepthFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

		commHelper.endSingleTimeCommands(commandBuffer, pools.command, queues.graphics);

		processMesh(model);

		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
		std::cout << "done!\n\n\n";
	}

	void createImageViews() {
		swapchain.imageViews.resize(swapchain.images.size());
		for (size_t i = 0; i < swapchain.images.size(); i++) {

			auto imageCreateInfo = vk::ImageViewCreateInfo(
				vk::ImageViewCreateFlags(),
				vk::Image(swapchain.images[i]),
				vk::ImageViewType::e2D,
				vk::Format(swapchain.imageFormat),
				vk::ComponentMapping(),
				vk::ImageSubresourceRange(
					vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor),
					0, 1, 0, 1
				),
				nullptr
			);
			try {
				swapchain.imageViews[i] = device.createImageView(imageCreateInfo);
			}
			catch (vk::SystemError err) {
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

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		std::vector<VkDescriptorSetLayoutBinding> bindingsAll = {
			uboLayoutBinding
		};
		layoutInfo.bindingCount = static_cast<uint32_t>(bindingsAll.size());
		layoutInfo.pBindings = bindingsAll.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &renderState.dsl) != VK_SUCCESS) {
			throw std::runtime_error("failed to create full descriptor set layout!");
		}

	}

	void createGraphicsPipelines() {

		ShaderHelper vertShader;
		vertShader.init("vert", Type::VERT, device);
		vertShader.readCompiledSPIRVAndCreateShaderModule("../shaders/skeletal-anim/skel.vert.spv");

		ShaderHelper fragShader;
		fragShader.init("frag", Type::FRAG, device);
		fragShader.readCompiledSPIRVAndCreateShaderModule("../shaders/skeletal-anim/skel.frag.spv");

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

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

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
		rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
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
		pipelineLayoutInfo.pSetLayouts = &renderState.dsl;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &renderState.pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchain.imageFormat,
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
		pipelineInfo.layout = renderState.pipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
		pipelineInfo.pStages = shaderStages;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &renderState.graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShader.shaderModule, nullptr);
		vkDestroyShaderModule(device, fragShader.shaderModule, nullptr);
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

	void processMesh(Drawable drawable) {

		VkCommandPool tempPool;

		commMutex.lock();

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			QUEUE_TYPE_GRAPHICS,
			tempPool
		);

		commMutex.unlock();

		for (int j = 0; j < drawable.pos.size(); j++) {
			Vertex vertex{};

			vertex.pos = drawable.pos[j];
			vertex.joints = drawable.joints[j];
			vertex.weights = drawable.weights[j];

			drawData.vertices.push_back(vertex);
		}

		drawable.pos.clear();
		drawable.pos.shrink_to_fit();
		drawable.normals.clear();
		drawable.normals.shrink_to_fit();
		drawable.tangents.clear();
		drawable.tangents.shrink_to_fit();
		drawable.texCoords.clear();
		drawable.texCoords.shrink_to_fit();
		drawable.joints.clear();
		drawable.joints.shrink_to_fit();
		drawable.weights.clear();
		drawable.weights.shrink_to_fit();

		//create vertex buffer
		VkDeviceSize vertexBufferSize = sizeof(drawData.vertices[0]) * drawData.vertices.size();

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
		memcpy(vertexData, drawData.vertices.data(), (size_t)vertexBufferSize);
		vkUnmapMemory(device, stagingVertexBufferMemory);

		memHelper.createBuffer(
			vertexBufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			drawData.vertexBuffer,
			drawData.vertexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);
		VkCommandBuffer commandBuffer;
		commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

		memHelper.copyBuffer(commandBuffer, stagingVertexBuffer, drawData.vertexBuffer, vertexBufferSize);

		commMutex.lock();

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);	//kooky

		vkDestroyBuffer(device, stagingVertexBuffer, nullptr);
		vkFreeMemory(device, stagingVertexBufferMemory, nullptr);

		drawData.vertices.clear();
		drawData.vertices.shrink_to_fit();

		//create index buffer
		VkDeviceSize indexBufferSize = sizeof(drawable.indices[0]) * drawable.indices.size();
		drawData.indices = drawable.indices.size();
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
			drawData.indexBuffer,
			drawData.indexBufferMemory,
			QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
		);
		commandBuffer = commHelper.beginSingleTimeCommands(tempPool);

		memHelper.copyBuffer(commandBuffer, stagingIndexBuffer, drawData.indexBuffer, indexBufferSize);

		memMutex.unlock();

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);	//kooky 2

		commMutex.unlock();

		vkDestroyBuffer(device, stagingIndexBuffer, nullptr);
		vkFreeMemory(device, stagingIndexBufferMemory, nullptr);

		drawable.indices.clear();
		drawable.indices.shrink_to_fit();

		vkDestroyCommandPool(device, tempPool, nullptr);
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniform.buffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniform.buffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniform.buffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			memHelper.createBuffer(
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniform.buffers[i],
				uniform.buffersMemory[i],
				QUEUE_TYPE_GRAPHICS | QUEUE_TYPE_TRANSFER
			);
			vkMapMemory(device, uniform.buffersMemory[i], 0, bufferSize, 0, &uniform.buffersMapped[i]);
		}
	}
	void createDescriptorPool() {
		VkDescriptorPoolSize uniformPoolInfo = {
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 2
		};

		std::array<VkDescriptorPoolSize, 1> poolSizes = { uniformPoolInfo };

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.poolSizeCount = 1;
		poolInfo.maxSets = 2;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &renderState.descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets() {
		{
			std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, renderState.dsl);

			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = renderState.descriptorPool;
			allocInfo.descriptorSetCount = 2;
			allocInfo.pSetLayouts = layouts.data();
			renderState.descriptorSets.resize(layouts.size());

			VkResult allocResult = vkAllocateDescriptorSets(device, &allocInfo, renderState.descriptorSets.data());
			if (allocResult != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}
		}

		{
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				VkDescriptorBufferInfo uniformBufferInfo{};
				uniformBufferInfo.buffer = uniform.buffers[i];
				uniformBufferInfo.offset = 0;
				uniformBufferInfo.range = sizeof(UniformBufferObject);

				std::array<VkWriteDescriptorSet, 1> descWrites{};

				descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[0].dstSet = renderState.descriptorSets[i];
				descWrites[0].dstBinding = 0;
				descWrites[0].dstArrayElement = 0;
				descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descWrites[0].descriptorCount = 1;
				descWrites[0].pBufferInfo = &uniformBufferInfo;

				vkUpdateDescriptorSets(device, 1, descWrites.data(), 0, nullptr);
			}
		}
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = pools.command;
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
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderingAttachmentInfoKHR colorAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = swapchain.imageViews[imageIndex],
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = clearValues[0]
		};

		VkRenderingAttachmentInfoKHR depthAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = depthRender.imageView,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.clearValue = clearValues[1]
		};

		VkRenderingInfoKHR renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, swapchain.extent.width, swapchain.extent.height},
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
			.image = swapchain.images[imageIndex],
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
			.image = depthRender.image,
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
		viewport.width = static_cast<float>(swapchain.extent.width);
		viewport.height = static_cast<float>(swapchain.extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapchain.extent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		VkDeviceSize offsets[] = { 0 };

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderState.graphicsPipeline);

		VkBuffer vertexBuffers[] = { drawData.vertexBuffer };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, drawData.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderState.pipelineLayout, 0, 1, &renderState.descriptorSets[currentFrame], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(drawData.indices), 1, 0, 0, 0);

		vkCmdEndRendering(commandBuffer);

		VkImageMemoryBarrier colorImageEndTransitionBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapchain.images[imageIndex],
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

	glm::vec4 lerp(glm::vec4 x, glm::vec4 y, float t) {
		return x * (1.f - t) + y * t;
	}

	void updateUniformBuffer(uint32_t currentImage, float deltaTime) {
		camera.update(deltaTime);

		UniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);
		ubo.view = camera.getViewMatrix();
		ubo.proj = glm::perspective(glm::radians(45.0f), swapchain.extent.width / (float)swapchain.extent.height, 0.01f, 10.0f);
		ubo.proj[1][1] *= -1;
		ubo.camPos = camera.position;
		
		timeElapsed += deltaTime * scalingFactor;
		if(playAnim) for (auto& anim : model.skeleton.animations) {
			if (anim.type == AnimType::ROT) {
				size_t timings = anim.keyframeTimings.size();
				std::cout << timeElapsed;
				std::cout << "\nTimings found: " << timings;
				size_t floorIndex = std::numeric_limits<size_t>::max();
				for (size_t j = 0; j < anim.keyframeTimings.size() - 1; j++) {
					if (anim.keyframeTimings[j] <= timeElapsed && anim.keyframeTimings[j+1] >= timeElapsed) {
						floorIndex = j;
					}
				}
				if (floorIndex >= (anim.keyframeTimings.size() - 1)) {
					timeElapsed = 0.0f;

					glm::vec4 rotAmount = std::get<std::vector<glm::vec4>>(anim.keyframeValues)[0];
					model.skeleton.joints[anim.jointIdx - 1].globalTransform = glm::toMat4(glm::quat(rotAmount[3], rotAmount[0], rotAmount[1], rotAmount[2])) * model.skeleton.joints[anim.jointIdx - 1].transform;
				}
				else if (floorIndex != 0) {
					glm::vec4 rotAmount_0 = std::get<std::vector<glm::vec4>>(anim.keyframeValues)[floorIndex];
					glm::vec4 rotAmount_1 = std::get<std::vector<glm::vec4>>(anim.keyframeValues)[floorIndex + 1];
					float interpValue = (timeElapsed - anim.keyframeTimings[floorIndex]) / (anim.keyframeTimings[floorIndex] - anim.keyframeTimings[floorIndex - 1]);
					glm::vec4 finalRot = lerp(rotAmount_0, rotAmount_1, interpValue);
					model.skeleton.joints[anim.jointIdx - 1].globalTransform = glm::toMat4(glm::quat(finalRot[3], finalRot[0], finalRot[1], finalRot[2])) * model.skeleton.joints[anim.jointIdx - 1].transform;
				}
				else {
					glm::vec4 rotAmount = std::get<std::vector<glm::vec4>>(anim.keyframeValues)[0];
					model.skeleton.joints[anim.jointIdx - 1].globalTransform = glm::toMat4(glm::quat(rotAmount[3], rotAmount[0], rotAmount[1], rotAmount[2])) * model.skeleton.joints[anim.jointIdx - 1].transform;
				}

				std::cout << " and timing selected: " << floorIndex <<"\n";
			}
		}
		for (int i = 0; i < 2; i++) {
			ubo.joints[i] = model.skeleton.invBindMatrices[i] * model.skeleton.joints[i].transform * model.skeleton.joints[i].globalTransform;
		}
		memcpy(uniform.buffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
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
		lastTime = latestTime;

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

		if (vkQueueSubmit(queues.graphics, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { swapchain.swapchain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(queues.present, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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

		createImageViews();
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationlayers) {
			bool layerFound = true;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	void cleanupSwapChain() {
		vkDestroyImageView(device, colorRender.imageView, nullptr);
		vkDestroyImage(device, colorRender.image, nullptr);
		vkFreeMemory(device, colorRender.imageMemory, nullptr);
		vkDestroyImageView(device, depthRender.imageView, nullptr);
		vkDestroyImage(device, depthRender.image, nullptr);
		vkFreeMemory(device, depthRender.imageMemory, nullptr);

		for (auto swapChainImageView : swapchain.imageViews) {
			vkDestroyImageView(device, swapChainImageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);
	}

	void cleanup() {
		cleanupSwapChain();

		vkDestroyBuffer(device, drawData.indexBuffer, nullptr);
		vkFreeMemory(device, drawData.indexBufferMemory, nullptr);

		vkDestroyBuffer(device, drawData.vertexBuffer, nullptr);
		vkFreeMemory(device, drawData.vertexBufferMemory, nullptr);

		vkDestroyDescriptorSetLayout(device, renderState.dsl, nullptr);

		vkDestroyDescriptorPool(device, renderState.descriptorPool, nullptr);

		vkDestroyPipeline(device, renderState.graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, renderState.pipelineLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, pools.command, nullptr);

		vkDestroyDevice(device, nullptr);

		vulkanInit.destroy();

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main() {
	try {
		GLTFParser animParser;
		animParser.parse("../models/SimpleSkin.gltf");
		SkeletalAnimTest app(640, 480, animParser.drawables[0]);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}