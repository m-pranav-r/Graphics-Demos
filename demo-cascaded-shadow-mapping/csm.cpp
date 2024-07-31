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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#define SHADOW_MAP_SIZE 1024
#define SHADOW_MAP_FORMAT VK_FORMAT_D32_SFLOAT

#define NEAR_PLANE 0.01f
#define FAR_PLANE  10.0f
#define FOV 45.0f

#ifdef _DEBUG
bool isDebugEnv = true;
#else
bool isDebugEnv = false;
#endif

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 tangent;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, tangent);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 light;
	glm::vec3 camPos;	float _pad0;
	glm::vec3 lightPos;	float _pad1;
};

struct ShadowMapData {
	glm::mat4 model;
	glm::mat4 viewProj;
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
	VkImage baseColorImage, metallicRoughnessImage, normalImage;
	VkImageView baseColorImageView, metallicRoughnessImageView, normalImageView;
	VkSampler baseColorSampler, metallicRoughnessSampler, normalSampler;
	VkDeviceMemory baseColorMemory, metallicRoughnessMemory, normalMemory;
	std::vector<Vertex> vertices;
	VkBuffer vertexBuffer, indexBuffer;
	VkDeviceMemory vertexBufferMemory, indexBufferMemory;
	size_t indices;

	bool isAlphaModeMask = false, hasNormal = false, hasMR = false, hasTangents = false;
};

struct {
	bool focusOnApp = true;
} controls;

Camera camera;

class CSMDemo {
public:
	CSMDemo(uint32_t width, uint32_t height, std::vector<Drawable>& drawables) : WIDTH(width), HEIGHT(height), drawables(drawables) {};

	void run() {
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	uint32_t WIDTH, HEIGHT;
	std::vector<Drawable> drawables;

	Init vulkanInit;
	DeviceHelper deviceHelper;
	MemoryHelper memHelper;			std::mutex memMutex;
	CommandHelper commHelper;		std::mutex commMutex;

	struct {
		VkInstance instance;
		VkSurfaceKHR surface;
		GLFWwindow* window;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		vk::Device device;
	} app;
	
	struct {
		VkQueue graphics;
		VkQueue present;
		VkQueue transfer;
	} queues;

	struct {
		VkSwapchainKHR swapChain;
		std::vector<VkImage> images;
		VkFormat imageFormat;
		VkExtent2D extent;
		std::vector<VkImageView> imageViews;
	} swapchain;

	struct {
		VkPipelineLayout pipelineLayout;
		VkDescriptorSetLayout fullDSL;
		VkDescriptorSetLayout shadowDSL;
		std::vector<VkDescriptorSet> descriptorSets;
		VkDescriptorSet shadowDescSet;
		VkPipeline noPCFPipeline, PCFPipeline;
		VkDescriptorPool descriptorPool;
	}model;

	VkCommandPool commandPool;
	VkCommandPool transferPool;
	uint32_t mipLevels;

	std::vector<DrawableHandle> drawableHandles;
	std::mutex drawableHandlesMutex;

	struct{
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
	} color, depth;

	struct {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
		VkSampler sampler;
	} shadow;

	struct {
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		VkFence fence;
	} shadowPass;

	struct {
		float 
			eye[3] = { -2.0f, 4.0f, -1.0f },
			point = 2.0f,
			near = NEAR_PLANE,
			far = FAR_PLANE
		;
		ShadowMapData data;
	} shadowVars;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkBuffer offscreenUniformBuffer;
	VkDeviceMemory offscreenUniformBufferMemory;
	void* offscreenUniformBufferMapped;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore >imageAvailableSemaphores;
	std::vector<VkSemaphore >renderFinishedSemaphores;
	std::vector<VkFence >inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;
	bool isShadowMapLatest = true;
	bool isShaderLatest = true;
	bool usePCF = false;

	std::chrono::steady_clock::time_point lastTime;

	glm::vec3 sponzaScaleMatrix = glm::vec3(0.000800000038);

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<CSMDemo*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {

		vulkanInit.isDebug = isDebugEnv;
		vulkanInit.width = WIDTH;
		vulkanInit.height = HEIGHT;
		vulkanInit.title = "Cascaded Shadow Mapping Demo";
		vulkanInit.addLayer("VK_LAYER_KHRONOS_validation");
		vulkanInit.addInstanceExtension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
		vulkanInit.setFramebufferResizeFunc(framebufferResizeCallback);
		vulkanInit.setCursorCallback(
			[](GLFWwindow* window, double x, double y) {
				ImGuiIO& io = ImGui::GetIO();
				if (controls.focusOnApp) camera.processGLFWMouseEvent(window, x, y);
				else camera.prevFrameXPos = -1;
			}
		);
		vulkanInit.setMouseButtonCallback(
			[](GLFWwindow* window, int button, int action, int mods) {
				ImGuiIO& io = ImGui::GetIO();
				io.AddMouseButtonEvent(button, action == GLFW_PRESS);
				if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
					if (io.WantCaptureMouse || controls.focusOnApp) {
						io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;
						io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
						glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
						glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
						controls.focusOnApp = false;
					}
					else {
						io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableSetMousePos;
						io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
						glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
						glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
						controls.focusOnApp = true;
					}
				}
			}
		);
		vulkanInit.setKeyboardCallback(
			[](GLFWwindow* window, int key, int scancode, int action, int mods) {
				camera.processGLFWKeyboardEvent(window, key, scancode, action, mods);
			}
		);
		vulkanInit.init();

		camera.velocity = glm::vec3(0.f);
		camera.position = glm::vec3(2.f);
		camera.pitch = 0;
		camera.yaw = 0;
		camera.scalingFactor = 1.0f;

		app.instance = vulkanInit.getInstance();
		app.window = vulkanInit.getWindow();
		app.surface = vulkanInit.getSurface();

		deviceHelper.instance = app.instance;
		deviceHelper.surface = app.surface;
		deviceHelper.addLayer("VK_LAYER_KHRONOS_validation");
		deviceHelper.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		deviceHelper.initDevices();
		deviceHelper.createSwapchain(swapchain.swapChain, app.window, swapchain.images, swapchain.imageFormat, swapchain.extent);
		app.device = deviceHelper.getDevice();
		app.physicalDevice = deviceHelper.getPhysicalDevice();

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
			commandPool
		);

		commHelper.createCommandPool(
			VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			QUEUE_TYPE_TRANSFER,
			transferPool
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
			color.image,
			color.memory
		);

		color.view = createImageView(color.image, swapchain.imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		memHelper.createImage(
			swapchain.extent.width,
			swapchain.extent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			deviceHelper.findDepthFormat(),
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depth.image,
			depth.memory
		);
		depth.view = createImageView(depth.image, deviceHelper.findDepthFormat(), VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(commandPool);

		memHelper.transitionImageLayout(commandBuffer, depth.image, deviceHelper.findDepthFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

		commHelper.endSingleTimeCommands(commandBuffer, commandPool, queues.graphics);

		processDrawables();

		createUniformBuffers();
		createDescriptorPool();

			//shadow mapping pass
			//attain a depth image to derive lighting from in this pass
			//one setup, one pass - static light for now
			createShadowImage();
			setupShadowPass();
			performShadowRender();

		createDescriptorSets();

			setupUI();

		createCommandBuffers();
			
		createSyncObjects();
		std::cout << "done!\n\n\n";
	}

	void createImageViews() {
		swapchain.imageViews.resize(swapchain.images.size());
		for (size_t i = 0; i < swapchain.images.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapchain.images[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapchain.imageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			VkResult swapChainImageViewCreateResult = vkCreateImageView(app.device, &createInfo, nullptr, &swapchain.imageViews[i]);
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

		VkDescriptorSetLayoutBinding metallicRoughnessLayoutBinding{};
		metallicRoughnessLayoutBinding.binding = 2;
		metallicRoughnessLayoutBinding.descriptorCount = 1;
		metallicRoughnessLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		metallicRoughnessLayoutBinding.pImmutableSamplers = nullptr;
		metallicRoughnessLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding normalLayoutBinding{};
		normalLayoutBinding.binding = 3;
		normalLayoutBinding.descriptorCount = 1;
		normalLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		normalLayoutBinding.pImmutableSamplers = nullptr;
		normalLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		std::vector<VkDescriptorSetLayoutBinding> bindingsAll = {
			uboLayoutBinding,
			baseColorLayoutBinding,
			metallicRoughnessLayoutBinding,
			normalLayoutBinding,
		};
		layoutInfo.bindingCount = static_cast<uint32_t>(bindingsAll.size());
		layoutInfo.pBindings = bindingsAll.data();

		if (vkCreateDescriptorSetLayout(app.device, &layoutInfo, nullptr, &model.fullDSL) != VK_SUCCESS) {
			throw std::runtime_error("failed to create full descriptor set layout!");
		}

		//layout for shadow-only desc set
		VkDescriptorSetLayoutBinding shadowLayoutBinding{};
		shadowLayoutBinding.binding = 0;
		shadowLayoutBinding.descriptorCount = 1;
		shadowLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		shadowLayoutBinding.pImmutableSamplers = nullptr;
		shadowLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &shadowLayoutBinding;

		if (vkCreateDescriptorSetLayout(app.device, &layoutInfo, nullptr, &model.shadowDSL) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow-only descriptor set layout!");
		}
	}

	void createGraphicsPipelines() {

		if (!isShaderLatest) {
			vkDestroyPipelineLayout(app.device, model.pipelineLayout, nullptr);
			vkDestroyPipeline(app.device, model.noPCFPipeline, nullptr);
			vkDestroyPipeline(app.device, model.PCFPipeline, nullptr);
		}

		ShaderHelper vertShader;
		vertShader.init("vert", Type::VERT, app.device);
		vertShader.readCompiledSPIRVAndCreateShaderModule("../shaders/csm/model.vert.spv");

		ShaderHelper fragShader;
		fragShader.init("frag", Type::FRAG, app.device);
		fragShader.readCompiledSPIRVAndCreateShaderModule("../shaders/csm/model.frag.spv");

		uint32_t enablePCF = 0;
		VkSpecializationMapEntry specialEntry{};
		specialEntry.constantID = 0;
		specialEntry.offset = 0;
		specialEntry.size = sizeof(uint32_t);

		VkSpecializationInfo specialInfo{};
		specialInfo.mapEntryCount = 1;	specialInfo.pMapEntries = &specialEntry;
		specialInfo.dataSize = sizeof(uint32_t);
		specialInfo.pData = &enablePCF;

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
		fragShaderStageInfo.pSpecializationInfo = &specialInfo;

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

		std::array<VkDescriptorSetLayout, 2> setLayouts = { model.shadowDSL, model.fullDSL };
		
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = setLayouts.size();
		pipelineLayoutInfo.pSetLayouts = setLayouts.data();

		if (vkCreatePipelineLayout(app.device, &pipelineLayoutInfo, nullptr, &model.pipelineLayout) != VK_SUCCESS) {
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
		pipelineInfo.layout = model.pipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
		pipelineInfo.pStages = shaderStages;

		if (vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &model.noPCFPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		enablePCF = 1;
		if (vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &model.PCFPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pcf graphics pipeline!");
		}

		vkDestroyShaderModule(app.device, vertShader.shaderModule, nullptr);
		vkDestroyShaderModule(app.device, fragShader.shaderModule, nullptr);

		isShaderLatest = true;
	}

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(app.physicalDevice, &physicalDeviceProperties);

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

		if (vkCreateImageView(app.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	void ProcessDrawable(Drawable drawable) {

		DrawableHandle currDrawableHandle;

		std::vector<Texture> textures = {
			drawable.mat.baseColorTex,
		};

		currDrawableHandle.isAlphaModeMask = drawable.mat.isAlphaModeMask;
		currDrawableHandle.hasMR = drawable.mat.hasMR;
		currDrawableHandle.hasNormal = drawable.mat.hasNormal;
		currDrawableHandle.hasTangents = drawable.hasTangents;

		if (currDrawableHandle.hasMR) textures.push_back(drawable.mat.metalRoughTex);
		else {
			return;
		}
		if (currDrawableHandle.hasNormal) textures.push_back(drawable.mat.normalTex);

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
			if (vkMapMemory(app.device, stagingBufferMemory, 0, imageSize, 0, &data) != VK_SUCCESS) {
				throw std::runtime_error("failed to map texture memory!");
			}
			memcpy(data, texture.pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(app.device, stagingBufferMemory);

			stbi_image_free(texture.pixels);

			VkImage* textureImage = nullptr;
			VkDeviceMemory* textureImageMemory = nullptr;
			VkImageView* textureImageView = nullptr;
			VkSampler* textureSampler = nullptr;
			VkFormat textureImageFormat;

			switch (texture.type) {
			case BASE:
			{
				textureImage = &currDrawableHandle.baseColorImage;
				textureImageMemory = &currDrawableHandle.baseColorMemory;
				textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
				textureImageView = &currDrawableHandle.baseColorImageView;
				textureSampler = &currDrawableHandle.baseColorSampler;
				break;
			}
			case METALLIC_ROUGHNESS: {
				textureImage = &currDrawableHandle.metallicRoughnessImage;
				textureImageMemory = &currDrawableHandle.metallicRoughnessMemory;
				textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
				textureImageView = &currDrawableHandle.metallicRoughnessImageView;
				textureSampler = &currDrawableHandle.metallicRoughnessSampler;
				break;
			}
			case NORMAL: {
				textureImage = &currDrawableHandle.normalImage;
				textureImageMemory = &currDrawableHandle.normalMemory;
				textureImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
				textureImageView = &currDrawableHandle.normalImageView;
				textureSampler = &currDrawableHandle.normalSampler;
				break;
			}
			}

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

			commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);

			vkDestroyBuffer(app.device, stagingBuffer, nullptr);
			vkFreeMemory(app.device, stagingBufferMemory, nullptr);

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

			commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);

			commMutex.unlock();

			*textureImageView = createImageView(*textureImage, textureImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;

			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(app.physicalDevice, &properties);

			samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.minLod = 0.0f;
			samplerInfo.maxLod = static_cast<float>(1);
			samplerInfo.mipLodBias = 0.0f;

			if (vkCreateSampler(app.device, &samplerInfo, nullptr, textureSampler) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture sampler!");
			}
		}
		for (int j = 0; j < drawable.pos.size(); j++) {
			Vertex vertex{};

			vertex.pos = drawable.pos[j];
			vertex.normal = drawable.normals[j];
			if (drawable.hasTangents)vertex.tangent = drawable.tangents[j];
			else vertex.tangent = glm::vec4(0.0);
			vertex.texCoord = drawable.texCoords[j];

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
		vkMapMemory(app.device, stagingVertexBufferMemory, 0, vertexBufferSize, 0, &vertexData);
		memcpy(vertexData, currDrawableHandle.vertices.data(), (size_t)vertexBufferSize);
		vkUnmapMemory(app.device, stagingVertexBufferMemory);

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

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);	//kooky

		vkDestroyBuffer(app.device, stagingVertexBuffer, nullptr);
		vkFreeMemory(app.device, stagingVertexBufferMemory, nullptr);

		currDrawableHandle.vertices.clear();
		currDrawableHandle.vertices.shrink_to_fit();

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
		vkMapMemory(app.device, stagingIndexBufferMemory, 0, indexBufferSize, 0, &indexData);
		memcpy(indexData, drawable.indices.data(), (size_t)indexBufferSize);
		vkUnmapMemory(app.device, stagingIndexBufferMemory);

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

		commHelper.endSingleTimeCommands(commandBuffer, tempPool, queues.graphics);

		commMutex.unlock();

		vkDestroyBuffer(app.device, stagingIndexBuffer, nullptr);
		vkFreeMemory(app.device, stagingIndexBufferMemory, nullptr);


		drawable.indices.clear();
		drawable.indices.shrink_to_fit();

		drawableHandlesMutex.lock();

		drawableHandles.push_back(currDrawableHandle);

		drawableHandlesMutex.unlock();

		vkDestroyCommandPool(app.device, tempPool, nullptr);
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

		vkDestroyCommandPool(app.device, transferPool, nullptr);

		drawables.clear();
		drawables.shrink_to_fit();

		std::cerr << "\nAll drawables processed.\n";
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
			vkMapMemory(app.device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
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
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

		if (vkCreateDescriptorPool(app.device, &poolInfo, nullptr, &model.descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets() {
		{
			std::vector<VkDescriptorSetLayout> layouts(drawableHandles.size() * MAX_FRAMES_IN_FLIGHT, model.fullDSL);

			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = model.descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
			allocInfo.pSetLayouts = layouts.data();
			model.descriptorSets.resize(layouts.size());

			if (vkAllocateDescriptorSets(app.device, &allocInfo, model.descriptorSets.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			allocInfo.descriptorSetCount = 1;
			allocInfo.pSetLayouts = &model.shadowDSL;

			if (vkAllocateDescriptorSets(app.device, &allocInfo, &model.shadowDescSet) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate shadow descriptor set!");
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

				VkDescriptorImageInfo metallicRoughnessImageInfo{};
				metallicRoughnessImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				metallicRoughnessImageInfo.imageView = drawableHandles[j].metallicRoughnessImageView;
				metallicRoughnessImageInfo.sampler = drawableHandles[j].metallicRoughnessSampler;

				VkDescriptorImageInfo normalImageInfo{};
				normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				normalImageInfo.imageView = drawableHandles[j].normalImageView;
				normalImageInfo.sampler = drawableHandles[j].normalSampler;

				std::array<VkWriteDescriptorSet, 4> descWrites{};

				descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[0].dstSet = model.descriptorSets[(2 * j) + i];
				descWrites[0].dstBinding = 0;
				descWrites[0].dstArrayElement = 0;
				descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descWrites[0].descriptorCount = 1;
				descWrites[0].pBufferInfo = &uniformBufferInfo;

				descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[1].dstSet = model.descriptorSets[(2 * j) + i];
				descWrites[1].dstBinding = 1;
				descWrites[1].dstArrayElement = 0;
				descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descWrites[1].descriptorCount = 1;
				descWrites[1].pImageInfo = &baseColorImageInfo;

				descWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[2].dstSet = model.descriptorSets[(2 * j) + i];
				descWrites[2].dstBinding = 2;
				descWrites[2].dstArrayElement = 0;
				descWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descWrites[2].descriptorCount = 1;
				descWrites[2].pImageInfo = &metallicRoughnessImageInfo;

				descWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descWrites[3].dstSet = model.descriptorSets[(2 * j) + i];
				descWrites[3].dstBinding = 3;
				descWrites[3].dstArrayElement = 0;
				descWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descWrites[3].descriptorCount = 1;
				descWrites[3].pImageInfo = &normalImageInfo;

				vkUpdateDescriptorSets(app.device, 4, descWrites.data(), 0, nullptr);
			}
		}

		VkDescriptorImageInfo shadowImageInfo{};
		shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		shadowImageInfo.imageView = shadow.view;
		shadowImageInfo.sampler = shadow.sampler;

		VkWriteDescriptorSet descWrite{};

		descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrite.dstSet = model.shadowDescSet;
		descWrite.dstBinding = 0;
		descWrite.dstArrayElement = 0;
		descWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descWrite.descriptorCount = 1;
		descWrite.pImageInfo = &shadowImageInfo;

		vkUpdateDescriptorSets(app.device, 1, &descWrite, 0, nullptr);
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(app.device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command buffers!");
		}
	}

	void createShadowImage() {
		memHelper.createImage(
			SHADOW_MAP_SIZE,
			SHADOW_MAP_SIZE,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			SHADOW_MAP_FORMAT,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			shadow.image,
			shadow.memory
		);
		shadow.view = createImageView(shadow.image, deviceHelper.findDepthFormat(), VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(commandPool);

			memHelper.transitionImageLayout(commandBuffer, shadow.image, deviceHelper.findDepthFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
		
		commHelper.endSingleTimeCommands(commandBuffer, commandPool, queues.graphics);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.mipLodBias = 0.0f;

		if (vkCreateSampler(app.device, &samplerInfo, nullptr, &shadow.sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow texture sampler!");
		}
	}

	void setupShadowPass() {
		ShaderHelper shadowMapVert;
		shadowMapVert.init("shadow vert", Type::VERT, app.device);
		shadowMapVert.readCompiledSPIRVAndCreateShaderModule("../shaders/csm/shadow.vert.spv");

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = shadowMapVert.shaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();

		VkVertexInputAttributeDescription attributeDescription{};
		attributeDescription.binding = 0;
		attributeDescription.location = 0;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = offsetof(Vertex, pos);

		VkVertexInputAttributeDescription attributeDescriptions[] = { attributeDescription };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
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
		rasterizer.depthClampEnable = VK_TRUE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.depthBiasEnable = VK_TRUE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 0;
		
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
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
		pipelineLayoutInfo.setLayoutCount = 0;

		VkPushConstantRange pushConstants;
		pushConstants.offset = 0;
		pushConstants.size = sizeof(ShadowMapData);
		pushConstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstants;

		if (vkCreatePipelineLayout(app.device, &pipelineLayoutInfo, nullptr, &shadowPass.pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 0,
			.depthAttachmentFormat = SHADOW_MAP_FORMAT
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = shadowPass.pipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.stageCount = 1;

		if (vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPass.pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow pipeline!");
		}

		vkDestroyShaderModule(app.device, shadowMapVert.shaderModule, nullptr);

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		vkCreateFence(app.device, &fenceInfo, nullptr, &shadowPass.fence);
	}

	void performShadowRender() {
		VkCommandBuffer shadowCmdBuf;

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(app.device, &allocInfo, &shadowCmdBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow command buffer!");
		}
		vkResetCommandBuffer(shadowCmdBuf, 0);
		
		//record command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;

		if (vkBeginCommandBuffer(shadowCmdBuf, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		if (!isShadowMapLatest) {
			VkImageMemoryBarrier colorImageStartTransitionBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
			};

			vkCmdPipelineBarrier(
				shadowCmdBuf,
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
		}

		std::array<VkClearValue, 1> clearValue{};
		clearValue[0].depthStencil = { 1.0f, 0 };

		VkRenderingAttachmentInfoKHR depthAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = shadow.view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = clearValue[0]
		};

		VkRenderingInfoKHR renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
			.layerCount = 1,
			.colorAttachmentCount = 0,
			.pDepthAttachment = &depthAttachment,
		};

		vkCmdBeginRendering(shadowCmdBuf, &renderingInfo);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(SHADOW_MAP_SIZE);
		viewport.height = static_cast<float>(SHADOW_MAP_SIZE);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(shadowCmdBuf, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = { SHADOW_MAP_SIZE , SHADOW_MAP_SIZE };
		vkCmdSetScissor(shadowCmdBuf, 0, 1, &scissor);

		VkDeviceSize offsets[] = { 0 };

		glm::mat4 shadowProj = glm::ortho(
			-shadowVars.point,
			 shadowVars.point,
			-shadowVars.point,
			 shadowVars.point,
			shadowVars.near,
			shadowVars.far
		);

		//if(shadowVars.projFlip) shadowProj[1][1] *= -1;

		glm::mat4 shadowView = glm::lookAt(
			glm::vec3(shadowVars.eye[0], shadowVars.eye[1], shadowVars.eye[2]),
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f)
		);
		
		shadowVars.data.model = glm::scale(glm::mat4(1.0f), sponzaScaleMatrix);
		shadowVars.data.viewProj = shadowProj * shadowView;

		vkCmdBindPipeline(shadowCmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPass.pipeline);
		//vkCmdSetDepthBias(shadowCmdBuf, 1.25f, 0.0f, 1.75f);
		vkCmdPushConstants(shadowCmdBuf, shadowPass.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowMapData), &shadowVars.data);

		//render model
		for (int i = 0; i < drawableHandles.size(); i++) {
			VkBuffer vertexBuffers[] = { drawableHandles[i].vertexBuffer };
			vkCmdBindVertexBuffers(shadowCmdBuf, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(shadowCmdBuf, drawableHandles[i].indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(shadowCmdBuf, static_cast<uint32_t>(drawableHandles[i].indices), 1, 0, 0, 0);
		}

		vkCmdEndRendering(shadowCmdBuf);

		VkImageMemoryBarrier colorImageEndTransitionBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			shadowCmdBuf,
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

		if (vkEndCommandBuffer(shadowCmdBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to record shadow command buffer!");
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &shadowCmdBuf;

		vkResetFences(app.device, 1, &shadowPass.fence);

		if (vkQueueSubmit(queues.graphics, 1, &submitInfo, shadowPass.fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffer!");
		}

		vkWaitForFences(app.device, 1, &shadowPass.fence, VK_TRUE, UINT64_MAX);

		isShadowMapLatest = true;
	}

	static void imgui_check_vk_result(VkResult err)
	{
		if (err == 0)
			return;
		fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
		if (err < 0)
			abort();
	}

	void setupUI() {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchain.imageFormat,
			.depthAttachmentFormat = deviceHelper.findDepthFormat()
		};

		ImGui_ImplGlfw_InitForVulkan(app.window, true);
		ImGui_ImplVulkan_InitInfo uiInfo{};
		uiInfo.Instance = app.instance;
		uiInfo.PhysicalDevice = app.physicalDevice;
		uiInfo.Device = app.device;
		uiInfo.QueueFamily = deviceHelper.getQueueFamilyIndices().graphicsFamily.value();
		uiInfo.Queue = queues.graphics;
		uiInfo.PipelineCache = nullptr;
		uiInfo.DescriptorPool = model.descriptorPool;
		uiInfo.UseDynamicRendering = true;
		uiInfo.Subpass = 0;
		uiInfo.MinImageCount = 2;
		uiInfo.ImageCount = 2;
		uiInfo.Allocator = nullptr;
		uiInfo.PipelineRenderingCreateInfo = pipelineRenderingCreateInfo;
		uiInfo.CheckVkResultFn = imgui_check_vk_result;
		ImGui_ImplVulkan_Init(&uiInfo);
	}

	void recordCommandBuffer(vk::CommandBuffer cmdBuf, uint32_t imageIndex) {
		vk::CommandBufferBeginInfo beginInfo{};

		if (cmdBuf.begin(&beginInfo) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { {0.4f, 0.5f, 0.6f, 1.0f} };
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
			.imageView = depth.view,
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
			cmdBuf,
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
			.image = depth.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			cmdBuf,
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

		vkCmdBeginRendering(cmdBuf, &renderingInfo);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapchain.extent.width);
		viewport.height = static_cast<float>(swapchain.extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapchain.extent;
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

		VkDeviceSize offsets[] = { 0 };

		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, usePCF ? model.PCFPipeline : model.noPCFPipeline);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, model.pipelineLayout, 0, 1, &model.shadowDescSet, 0, nullptr);

		//render model
		for (int i = 0; i < drawableHandles.size(); i++) {
			VkBuffer vertexBuffers[] = { drawableHandles[i].vertexBuffer };
			vkCmdBindVertexBuffers(cmdBuf, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(cmdBuf, drawableHandles[i].indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, model.pipelineLayout, 1, 1, &model.descriptorSets[(2 * i) + currentFrame], 0, nullptr);
			vkCmdDrawIndexed(cmdBuf, static_cast<uint32_t>(drawableHandles[i].indices), 1, 0, 0, 0);
		}

		recordUI(cmdBuf);

		vkCmdEndRendering(cmdBuf);

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
			cmdBuf,
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

		if (vkEndCommandBuffer(cmdBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void recordUI(VkCommandBuffer commandBuffer) {
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		/*
		ImGui::ShowDemoWindow();
		*/

		//add ui objects
		ImGui::Begin("Demo - Cascaded Shadow Mapping");

		//ImGui::PlotLines("Frametime Graph", graphValues, FRAMETIME_GRAPH_SIZE);
			ImGui::InputFloat3("Shadow Eye", shadowVars.eye);
			ImGui::InputFloat("Shadow Point", &shadowVars.point);
			ImGui::InputFloat("Shadow Near", &shadowVars.near);
			ImGui::InputFloat("Shadow Far", &shadowVars.far);
			ImGui::Checkbox("Enable PCF", &usePCF);

			if (ImGui::Button("Re-render Shadow Map"))
				isShadowMapLatest = false;
			if (ImGui::Button("Re-compile Shader"))
				isShaderLatest = false;

			ImGui::Image(model.shadowDescSet, ImVec2(SHADOW_MAP_SIZE * 0.5, SHADOW_MAP_SIZE * 0.5));

			if (ImGui::Button("Exit"))
				glfwSetWindowShouldClose(app.window, true);


		ImGui::End();

		ImGui::Render();
		ImDrawData* drawData = ImGui::GetDrawData();
		//render using the data provided in drawData
		ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
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
			if (vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(app.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create sync objects for a frame!");
			}
		}
	}

	void drawFrame() {
		vkWaitForFences(app.device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(app.device, swapchain.swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		vkResetFences(app.device, 1, &inFlightFences[currentFrame]);

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

		VkSwapchainKHR swapchains[] = { swapchain.swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(queues.graphics, &presentInfo);

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
		ubo.model = shadowVars.data.model;
		ubo.view = camera.getViewMatrix();
		ubo.proj = glm::perspective(glm::radians(FOV), swapchain.extent.width / (float)swapchain.extent.height, NEAR_PLANE, FAR_PLANE);
		ubo.proj[1][1] *= -1;
		ubo.light = shadowVars.data.viewProj;
		ubo.camPos = camera.position;
		ubo.lightPos = glm::vec3(shadowVars.eye[0], shadowVars.eye[1], shadowVars.eye[2]);

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(app.window)) {
			if (!isShadowMapLatest) {
				performShadowRender();
			}
			if (!isShaderLatest) {
				createGraphicsPipelines();
			}
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(app.device);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(app.window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(app.window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(app.device);

		cleanupSwapChain();

		deviceHelper.createSwapchain(swapchain.swapChain, app.window, swapchain.images, swapchain.imageFormat, swapchain.extent);
		//add routine here to recreate device swapchain
		createImageViews();
		
		memHelper.createImage(
			swapchain.extent.width,
			swapchain.extent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			swapchain.imageFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			color.image,
			color.memory
		);

		color.view = createImageView(color.image, swapchain.imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		memHelper.createImage(
			swapchain.extent.width,
			swapchain.extent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			deviceHelper.findDepthFormat(),
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depth.image,
			depth.memory
		);
		depth.view = createImageView(depth.image, deviceHelper.findDepthFormat(), VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		VkCommandBuffer commandBuffer = commHelper.beginSingleTimeCommands(commandPool);

		memHelper.transitionImageLayout(commandBuffer, depth.image, deviceHelper.findDepthFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

		commHelper.endSingleTimeCommands(commandBuffer, commandPool, queues.graphics);
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
		vkDestroyImageView(app.device, color.view, nullptr);
		vkDestroyImage(app.device, color.image, nullptr);
		vkFreeMemory(app.device, color.memory, nullptr);
		vkDestroyImageView(app.device, depth.view, nullptr);
		vkDestroyImage(app.device, depth.image, nullptr);
		vkFreeMemory(app.device, depth.memory, nullptr);

		for (auto swapChainImageView : swapchain.imageViews) {
			vkDestroyImageView(app.device, swapChainImageView, nullptr);
		}
		vkDestroySwapchainKHR(app.device, swapchain.swapChain, nullptr);
	}

	void destroyDrawableHandleData(DrawableHandle drawableHandle) {
		vkDestroySampler(app.device, drawableHandle.normalSampler, nullptr);
		vkDestroyImageView(app.device, drawableHandle.normalImageView, nullptr);
		vkDestroyImage(app.device, drawableHandle.normalImage, nullptr);
		vkFreeMemory(app.device, drawableHandle.normalMemory, nullptr);

		vkDestroySampler(app.device, drawableHandle.metallicRoughnessSampler, nullptr);
		vkDestroyImageView(app.device, drawableHandle.metallicRoughnessImageView, nullptr);
		vkDestroyImage(app.device, drawableHandle.metallicRoughnessImage, nullptr);
		vkFreeMemory(app.device, drawableHandle.metallicRoughnessMemory, nullptr);

		vkDestroySampler(app.device, drawableHandle.baseColorSampler, nullptr);
		vkDestroyImageView(app.device, drawableHandle.baseColorImageView, nullptr);
		vkDestroyImage(app.device, drawableHandle.baseColorImage, nullptr);
		vkFreeMemory(app.device, drawableHandle.baseColorMemory, nullptr);

		vkDestroyBuffer(app.device, drawableHandle.indexBuffer, nullptr);
		vkFreeMemory(app.device, drawableHandle.indexBufferMemory, nullptr);

		vkDestroyBuffer(app.device, drawableHandle.vertexBuffer, nullptr);
		vkFreeMemory(app.device, drawableHandle.vertexBufferMemory, nullptr);
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

		deleteTask.wait();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(app.device, uniformBuffers[i], nullptr);
			vkFreeMemory(app.device, uniformBuffersMemory[i], nullptr);
		}

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		vkDestroySampler(app.device, shadow.sampler, nullptr);
		vkDestroyImageView(app.device, shadow.view, nullptr);
		vkDestroyImage(app.device, shadow.image, nullptr);
		vkFreeMemory(app.device, shadow.memory, nullptr);

		vkDestroyDescriptorSetLayout(app.device, model.fullDSL, nullptr);
		vkDestroyDescriptorSetLayout(app.device, model.shadowDSL, nullptr);

		vkDestroyDescriptorPool(app.device, model.descriptorPool, nullptr);

		vkDestroyPipeline(app.device, model.noPCFPipeline, nullptr);
		vkDestroyPipeline(app.device, model.PCFPipeline, nullptr);
		vkDestroyPipeline(app.device, shadowPass.pipeline, nullptr);
		vkDestroyPipelineLayout(app.device, model.pipelineLayout, nullptr);
		vkDestroyPipelineLayout(app.device, shadowPass.pipelineLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(app.device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(app.device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(app.device, inFlightFences[i], nullptr);
		}
		vkDestroyFence(app.device, shadowPass.fence, nullptr);

		vkDestroyCommandPool(app.device, commandPool, nullptr);

		vkDestroyDevice(app.device, nullptr);

		vulkanInit.destroy();

		glfwDestroyWindow(app.window);
		glfwTerminate();
	}
};

int main() {
	try {
		GLTFParser sponzaParser;
		sponzaParser.parse_sponza("../models/Sponza/glTF/Sponza.gltf");
		CSMDemo app(1280, 720, sponzaParser.drawables);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}