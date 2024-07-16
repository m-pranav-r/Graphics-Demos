#include "vulkan/vulkan.h"

#include "device.h"

class CommandHelper {
public:
	void init(QueueFamilyIndices queueFamilyIndices, VkDevice device, VkPhysicalDevice physicalDevice);

	void createCommandPool(VkCommandPoolCreateFlags flags, int queueType, VkCommandPool& commandPool);

	VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool);

	void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkCommandPool commandPool, VkQueue queue);

private:

	QueueFamilyIndices primeDeviceIndices;

	VkDevice device;

	VkPhysicalDevice physicalDevice;
};