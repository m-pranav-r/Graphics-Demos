#pragma once

#include<iostream>
#include<vector>
#include<set>
#include<string>
#include<optional>
#include<algorithm>

#include<vulkan/vulkan.h>

#include<GLFW/glfw3.h>

#define QUEUE_TYPE_GRAPHICS		static_cast<int>( 1 << 0 )
#define QUEUE_TYPE_TRANSFER		static_cast<int>( 1 << 1 )

struct QueueFamilyIndices {
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> transferFamily;
	std::optional<uint32_t> graphicsFamily;

	bool isComplete();
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class DeviceHelper {
public:
	void initDevices();

	QueueFamilyIndices getQueueFamilyIndices();

	void createSwapchain(VkSwapchainKHR &swapchain, GLFWwindow *window, std::vector<VkImage>& swapChainImages, VkFormat& swapChainImageFormat, VkExtent2D& swapChainExtent);

	void addDeviceExtension(const char* extension);

	void addLayer(const char* layer);

	void addDeviceExtensions(std::vector<const char*> extensions);

	VkFormat findDepthFormat();

	VkDevice getDevice();

	VkPhysicalDevice getPhysicalDevice();

	void getQueues(VkQueue& graphicsQueue, VkQueue& presentQueue, VkQueue& transferQueue);

	bool isDebug = false;

	VkInstance instance;	//set instance!!
	VkSurfaceKHR surface;	//set surface!!
private:
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

	bool checkDeviceExtensionSupport(VkPhysicalDevice device);

	bool isDeviceSuitable(VkPhysicalDevice device);

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkSurfaceFormatKHR chooseSwapSurfaceFormatHDR(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);

	std::vector<const char*> deviceExtensions;

	std::vector<const char*> layers;
	
	QueueFamilyIndices primeDeviceIndices;

	VkPhysicalDevice physicalDevice;

	VkDevice device;
};