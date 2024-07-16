#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class Init {
public:

	void addLayer(const char* layer);

	void addInstanceExtension(const char* extension);

	void init();

	VkInstance getInstance();

	GLFWwindow* getWindow();

	VkSurfaceKHR getSurface();

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugUtilsMessengerEXT* pDebugMessenger);

	void destroyDebugMessenger();

	void setFramebufferResizeFunc(GLFWframebuffersizefun func);

	void setCursorCallback(GLFWcursorposfun func);

	void setKeyboardCallback(GLFWkeyfun func);

	void destroy();

	bool isDebug = false;
	const char* title;
	uint32_t width = -1, height = -1;

private:

	std::vector<const char*> getRequiredExtensions();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData);

	bool checkValidationLayerSupport();

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	void DestroyDebugUtilsMessengerEXT(VkInstance instance,
		VkDebugUtilsMessengerEXT debugMessenger,
		const VkAllocationCallbacks* pAllocator);

	std::vector<const char*> layers;
	std::vector<const char*> deviceExtensions;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	GLFWwindow* window;
	GLFWframebuffersizefun framebufferResizeCallback;
	GLFWcursorposfun cursorPositionCallback;
	GLFWkeyfun keyCallback;
};