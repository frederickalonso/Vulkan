#define _CRT_SECURE_NO_WARNINGS
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <optional>
#include <set>
#include <fstream>
#include <sstream>
#include <ctime>
#include <string>
#include <cstring>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Global Vulkan handles
GLFWwindow* window = nullptr;
VkInstance instance = VK_NULL_HANDLE;
VkDevice device;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkRenderPass renderPass;
VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

// Add after the global variables
std::vector<VkFramebuffer> swapChainFramebuffers;
std::vector<VkImageView> swapChainImageViews;
std::vector<VkImage> swapChainImages;
VkCommandPool commandPool;
VkCommandBuffer commandBuffer;
VkSemaphore imageAvailableSemaphore;
VkSemaphore renderFinishedSemaphore;
VkFence inFlightFence;

// Add after the device declaration
VkQueue graphicsQueue;
VkQueue presentQueue;
VkSwapchainKHR swapchain;
VkSurfaceKHR surface;

// Add this global variable to store the surface format
VkSurfaceFormatKHR globalSurfaceFormat;

// Global debug file handle for crash-safe logging
HANDLE debugFileHandle = INVALID_HANDLE_VALUE;

// Forward declarations
void logMessage(const std::string& message, bool isError = false);
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 VkBuffer& buffer, VkDeviceMemory& bufferMemory);
void createVertexBuffer();
void createIndexBuffer();
void createUniformBuffer();
void createDescriptorSetLayout();
void createDescriptorPool();
void createDescriptorSet();
void updateUniformBuffer();
void createRenderPass();
void createGraphicsPipeline();
void createImageViews();
void createFramebuffers();
void createCommandPool();
void createCommandBuffer();
void createSyncObjects();
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void drawFrame();
void writeDebugOutput(const char* message);
LONG WINAPI CustomUnhandledExceptionFilter(EXCEPTION_POINTERS* exceptionInfo);

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;

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
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        
        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

void loadModel() {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    
    std::string filepath = "SimpleCube.obj";
    std::ifstream file(filepath);
    if (!file.is_open()) {
        // Try alternative path as fallback
        filepath = "../SimpleCube.obj";
        file.open(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open model file: " + filepath);
        }
    }
    logMessage("Loading model from: " + filepath);
    
    // Clear any existing data
    vertices.clear();
    indices.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "v") {
            glm::vec3 pos;
            iss >> pos.x >> pos.y >> pos.z;
            positions.push_back(pos);
            logMessage("Added vertex position: " + std::to_string(pos.x) + ", " + 
                      std::to_string(pos.y) + ", " + std::to_string(pos.z));
        }
        else if (type == "vn") {
            glm::vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (type == "vt") {
            glm::vec2 texCoord;
            iss >> texCoord.x >> texCoord.y;
            texCoords.push_back(texCoord);
        }
        else if (type == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            
            auto processVertex = [&](const std::string& vertex) {
                std::istringstream viss(vertex);
                std::string indexStr;
                std::vector<int> indices;
                
                while (std::getline(viss, indexStr, '/')) {
                    indices.push_back(indexStr.empty() ? 0 : std::stoi(indexStr));
                }
                
                Vertex v{};
                if (!indices.empty() && indices[0] > 0 && indices[0] <= positions.size()) {
                    v.pos = positions[indices[0] - 1];
                } else {
                    logMessage("Warning: Invalid position index in face", true);
                }
                if (indices.size() > 1 && indices[1] > 0 && indices[1] <= texCoords.size()) {
                    v.texCoord = texCoords[indices[1] - 1];
                } else {
                    v.texCoord = glm::vec2(0.0f); // Default UV
                }
                if (indices.size() > 2 && indices[2] > 0 && indices[2] <= normals.size()) {
                    v.normal = normals[indices[2] - 1];
                } else {
                    v.normal = glm::vec3(0.0f, 1.0f, 0.0f); // Default normal
                }
                return v;
            };
            
            vertices.push_back(processVertex(v1));
            vertices.push_back(processVertex(v2));
            vertices.push_back(processVertex(v3));
            
            uint32_t baseIndex = static_cast<uint32_t>(vertices.size()) - 3;
            indices.push_back(baseIndex);
            indices.push_back(baseIndex + 1);
            indices.push_back(baseIndex + 2);
            
            logMessage("Added face with indices: " + std::to_string(baseIndex) + ", " + 
                      std::to_string(baseIndex + 1) + ", " + std::to_string(baseIndex + 2));
        }
    }
    
    if (vertices.empty() || indices.empty()) {
        throw std::runtime_error("Failed to load model: No vertices or indices found");
    }
    
    logMessage("Loaded model with " + std::to_string(vertices.size()) + " vertices and " + 
               std::to_string(indices.size()) + " indices");
    
    // Print first vertex for debugging
    if (!vertices.empty()) {
        const auto& v = vertices[0];
        logMessage("First vertex position: " + 
                  std::to_string(v.pos.x) + ", " + 
                  std::to_string(v.pos.y) + ", " + 
                  std::to_string(v.pos.z));
    }
}

// Debug logging function implementation
void logMessage(const std::string& message, bool isError) {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time);
    timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove newline

    std::string logEntry = timestamp + " [" + (isError ? "ERROR" : "INFO") + "] " + message;
    
    // Immediate console output
    std::cerr << logEntry << std::endl;
    std::cerr.flush(); // Force immediate output

    // File logging
    try {
        std::ofstream logFile("vulkan_log.txt", std::ios::app);
        if (logFile.is_open()) {
            logFile << logEntry << std::endl;
            logFile.flush(); // Force write to disk
            logFile.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to write to log file: " << e.what() << std::endl;
    }
}

// Validation layer support
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Add these constants for swapchain support
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Replace the instanceExtensions array with a function
std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// Replace the global function pointer declarations with:
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Replace the debug region functions with:
void beginDebugRegion(VkCommandBuffer cmdBuffer, const char* name, float r, float g, float b) {
    static auto pfn_vkCmdBeginDebugUtilsLabelEXT = 
        (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT");
    if (pfn_vkCmdBeginDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name;
        label.color[0] = r;
        label.color[1] = g;
        label.color[2] = b;
        label.color[3] = 1.0f;
        pfn_vkCmdBeginDebugUtilsLabelEXT(cmdBuffer, &label);
    }
}

void endDebugRegion(VkCommandBuffer cmdBuffer) {
    static auto pfn_vkCmdEndDebugUtilsLabelEXT = 
        (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT");
    if (pfn_vkCmdEndDebugUtilsLabelEXT) {
        pfn_vkCmdEndDebugUtilsLabelEXT(cmdBuffer);
    }
}

void setObjectName(VkDevice device, uint64_t object, VkObjectType objectType, const char* name) {
    static auto pfn_vkSetDebugUtilsObjectNameEXT = 
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT");
    if (pfn_vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = object;
        nameInfo.pObjectName = name;
        pfn_vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
}

// Helper struct for queue family indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

// Helper function to find queue families
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    if (surface == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot find queue families with null surface!");
    }

    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    
    logMessage("Searching through " + std::to_string(queueFamilyCount) + " queue families");

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
            logMessage("Found graphics queue family at index " + std::to_string(i));
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
            logMessage("Found present queue family at index " + std::to_string(i));
        }
        if (indices.isComplete()) break;
    }
    return indices;
}

// Add this to the device suitability check
bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

// Helper struct for swapchain support details
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    if (surface == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot query swap chain support with null surface!");
    }

    SwapChainSupportDetails details;
    
    // Get surface capabilities
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get surface capabilities!");
    }
    
    // Get surface formats
    uint32_t formatCount;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get surface format count!");
    }
    
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to get surface formats!");
        }
        logMessage("Found " + std::to_string(formatCount) + " surface formats");
    }
    
    // Get present modes
    uint32_t presentModeCount;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get present mode count!");
    }
    
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to get present modes!");
        }
        logMessage("Found " + std::to_string(presentModeCount) + " present modes");
    }
    
    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    // First try to find a format that supports both SRGB and UNORM
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            logMessage("Selected swap chain format: VK_FORMAT_B8G8R8A8_UNORM");
            globalSurfaceFormat = availableFormat;
            return availableFormat;
        }
    }

    // If not found, just return the first format
    logMessage("Using first available swap chain format: " + std::to_string(availableFormats[0].format));
    globalSurfaceFormat = availableFormats[0];
    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = { WIDTH, HEIGHT };
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

// Modify the isDeviceSuitable function
bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = findQueueFamilies(device, surface);
    
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    logMessage("Checking device: " + std::string(deviceProperties.deviceName));
    
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    if (!extensionsSupported) {
        logMessage("Device does not support required extensions", true);
        return false;
    }
    logMessage("Device supports all required extensions");

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        if (!swapChainAdequate) {
            logMessage("Device does not have adequate swap chain support", true);
        } else {
            logMessage("Device has adequate swap chain support");
            logMessage("Available formats: " + std::to_string(swapChainSupport.formats.size()));
            logMessage("Available present modes: " + std::to_string(swapChainSupport.presentModes.size()));
        }
    }
    
    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;
VkBuffer uniformBuffer;
VkDeviceMemory uniformBufferMemory;
VkDescriptorPool descriptorPool;
VkDescriptorSet descriptorSet;
VkDescriptorSetLayout descriptorSetLayout;

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void createVertexBuffer() {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    if (bufferSize == 0) {
        throw std::runtime_error("Cannot create vertex buffer with size 0");
    }

    createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBuffer, vertexBufferMemory);

    void* data;
    vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, vertexBufferMemory);
    
    logMessage("Created vertex buffer of size: " + std::to_string(bufferSize) + " bytes");
}

void createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    if (bufferSize == 0) {
        throw std::runtime_error("Cannot create index buffer with size 0");
    }

    createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBuffer, indexBufferMemory);

    void* data;
    vkMapMemory(device, indexBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, indexBufferMemory);
    
    logMessage("Created index buffer of size: " + std::to_string(bufferSize) + " bytes");
}

void createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffer, uniformBufferMemory);
}

void updateUniformBuffer() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>
        (currentTime - startTime).count();

    UniformBufferObject ubo{};
    // Rotate model around Y axis
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Move camera back and up slightly for better view
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                          glm::vec3(0.0f, 0.0f, 0.0f),
                          glm::vec3(0.0f, 1.0f, 0.0f));
                          
    ubo.proj = glm::perspective(glm::radians(45.0f),
                               WIDTH / (float)HEIGHT, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;  // Flip Y coordinate for Vulkan

    void* data;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBufferMemory);
}

void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void createDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

void createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = globalSurfaceFormat.format;  // Use the same format
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass!");
    }
    logMessage("Render pass created successfully");
}

void createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)WIDTH;
    viewport.height = (float)HEIGHT;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {WIDTH, HEIGHT};

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

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
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline!");
    }

    setObjectName(device, (uint64_t)graphicsPipeline, VK_OBJECT_TYPE_PIPELINE, "Main Graphics Pipeline");

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    
    logMessage("Graphics pipeline created successfully");
}

void createImageViews() {
    uint32_t imageCount;
    VkResult result = vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get swap chain images count!");
    }
    
    swapChainImages.resize(imageCount);
    result = vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapChainImages.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get swap chain images!");
    }

    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = globalSurfaceFormat.format;  // Use the same format as the swapchain
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        result = vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view " + std::to_string(i) + "! Error: " + std::to_string(result));
        }
    }
    logMessage("Created " + std::to_string(swapChainImageViews.size()) + " swap chain image views");
}

void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = WIDTH;
        framebufferInfo.height = HEIGHT;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer!");
        }
    }
    logMessage("Created " + std::to_string(swapChainFramebuffers.size()) + " framebuffers");
}

void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice, surface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }
    logMessage("Command pool created successfully");
}

void createCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }
    logMessage("Command buffer allocated successfully");
}

void createSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create synchronization objects!");
    }
    logMessage("Synchronization objects created successfully");
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }

    beginDebugRegion(commandBuffer, "Main Render Pass", 0.0f, 1.0f, 0.0f);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {WIDTH, HEIGHT};

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.4f, 1.0f}}}; // Blue background
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    beginDebugRegion(commandBuffer, "Draw Cube", 1.0f, 1.0f, 0.0f);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    endDebugRegion(commandBuffer);

    vkCmdEndRenderPass(commandBuffer);
    endDebugRegion(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

void drawFrame() {
    try {
        VkResult result;
        logMessage("DrawFrame: Starting new frame");
        
        // Wait for previous frame to finish
        logMessage("DrawFrame: Waiting for previous frame");
        result = vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to wait for fence: " + std::to_string(result));
        }
        
        result = vkResetFences(device, 1, &inFlightFence);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to reset fence: " + std::to_string(result));
        }
        logMessage("DrawFrame: Previous frame sync complete");

        // Acquire next image
        uint32_t imageIndex;
        logMessage("DrawFrame: Acquiring next swapchain image");
        result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            logMessage("DrawFrame: Swapchain out of date, recreating...", false);
            // Handle swapchain recreation here if needed
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image: " + std::to_string(result));
        }
        logMessage("DrawFrame: Acquired image index " + std::to_string(imageIndex));

        // Record command buffer
        logMessage("DrawFrame: Resetting command buffer");
        result = vkResetCommandBuffer(commandBuffer, 0);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to reset command buffer: " + std::to_string(result));
        }
        
        logMessage("DrawFrame: Recording command buffer");
        recordCommandBuffer(commandBuffer, imageIndex);
        logMessage("DrawFrame: Command buffer recorded");

        // Submit command buffer
        logMessage("DrawFrame: Submitting command buffer");
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer: " + std::to_string(result));
        }
        logMessage("DrawFrame: Command buffer submitted");

        // Present rendered image
        logMessage("DrawFrame: Presenting frame");
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapchain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            logMessage("DrawFrame: Swapchain out of date or suboptimal, recreating...", false);
            // Handle swapchain recreation here if needed
            return;
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image: " + std::to_string(result));
        }
        logMessage("DrawFrame: Frame presented successfully");
    } catch (const std::exception& e) {
        logMessage("Exception in drawFrame: " + std::string(e.what()), true);
        throw; // Re-throw to handle cleanup
    }
}

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
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

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    std::string severity;
    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: severity = "VERBOSE"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: severity = "INFO"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: severity = "WARNING"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: severity = "ERROR"; break;
        default: severity = "UNKNOWN"; break;
    }

    std::string type;
    switch (messageType) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: type = "GENERAL"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: type = "VALIDATION"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: type = "PERFORMANCE"; break;
        default: type = "UNKNOWN"; break;
    }

    std::string message = "[" + severity + "][" + type + "] " + pCallbackData->pMessage;
    if (pCallbackData->queueLabelCount > 0) {
        message += "\n  Queue Label: " + std::string(pCallbackData->pQueueLabels[0].pLabelName);
    }
    if (pCallbackData->cmdBufLabelCount > 0) {
        message += "\n  Command Buffer Label: " + std::string(pCallbackData->pCmdBufLabels[0].pLabelName);
    }
    if (pCallbackData->objectCount > 0) {
        message += "\n  Objects:";
        for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
            message += "\n    Type: " + std::to_string(pCallbackData->pObjects[i].objectType) +
                      ", Handle: " + std::to_string(pCallbackData->pObjects[i].objectHandle);
            if (pCallbackData->pObjects[i].pObjectName) {
                message += ", Name: " + std::string(pCallbackData->pObjects[i].pObjectName);
            }
        }
    }
    
    logMessage(message, messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
    return VK_FALSE;
}

void writeDebugOutput(const char* message) {
    // Write to debug output (visible in debugger)
    OutputDebugStringA(message);
    OutputDebugStringA("\n");
    
    // Write to our crash-safe file handle
    if (debugFileHandle != INVALID_HANDLE_VALUE) {
        DWORD written;
        DWORD messageLen = static_cast<DWORD>(strlen(message));
        WriteFile(debugFileHandle, message, messageLen, &written, NULL);
        WriteFile(debugFileHandle, "\n", 1, &written, NULL);
        FlushFileBuffers(debugFileHandle);
    }
}

LONG WINAPI CustomUnhandledExceptionFilter(EXCEPTION_POINTERS* exceptionInfo) {
    char buffer[1024];
    sprintf_s(buffer, "Crash detected! Exception code: 0x%08X at address: 0x%p\n",
        exceptionInfo->ExceptionRecord->ExceptionCode,
        exceptionInfo->ExceptionRecord->ExceptionAddress);
    writeDebugOutput(buffer);
    
    // Create minidump
    HANDLE dumpFile = CreateFileA("crash.dmp", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (dumpFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei;
        mei.ThreadId = GetCurrentThreadId();
        mei.ExceptionPointers = exceptionInfo;
        mei.ClientPointers = FALSE;
        
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), dumpFile,
            MiniDumpNormal, &mei, NULL, NULL);
        CloseHandle(dumpFile);
    }
    
    return EXCEPTION_CONTINUE_SEARCH;
}

int main() {
    // Set up crash handling
    SetUnhandledExceptionFilter(CustomUnhandledExceptionFilter);
    
    // Open debug output file
    debugFileHandle = CreateFileA(
        "vulkan_debug.txt",
        GENERIC_WRITE,
        FILE_SHARE_READ,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    writeDebugOutput("Application starting...");
    
    try {
        writeDebugOutput("Initializing GLFW...");
        if (!glfwInit()) {
            writeDebugOutput("Failed to initialize GLFW!");
            throw std::runtime_error("Failed to initialize GLFW!");
        }
        writeDebugOutput("GLFW initialized successfully");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        
        writeDebugOutput("Creating window...");
        window = glfwCreateWindow(WIDTH, HEIGHT, "MotorLaps Vulkan Viewport", nullptr, nullptr);
        if (!window) {
            writeDebugOutput("Failed to create GLFW window!");
            throw std::runtime_error("Failed to create GLFW window!");
        }
        writeDebugOutput("Window created successfully");

        // Load the model
        writeDebugOutput("Loading model...");
        loadModel();
        writeDebugOutput("Model loaded successfully");

        // Initialize Vulkan
        writeDebugOutput("Checking validation layers...");
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            writeDebugOutput("Validation layers not available!");
            throw std::runtime_error("Validation layers requested, but not available!");
        }
        writeDebugOutput("Validation layers checked successfully");

        // Create instance
        writeDebugOutput("Creating Vulkan instance...");
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Motorlaps";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Simulation";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        instanceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instanceInfo.ppEnabledExtensionNames = extensions.data();

        logMessage("Creating Vulkan instance with " + std::to_string(extensions.size()) + " extensions");
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
        logMessage("Vulkan instance created successfully");

        // Create surface
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
        logMessage("Window surface created successfully with handle: " + std::to_string((uint64_t)surface));

        // Make sure surface is valid before device creation
        if (surface == VK_NULL_HANDLE) {
            throw std::runtime_error("Surface handle is null before device creation!");
        }

        // Pick physical device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }
        logMessage("Found " + std::to_string(deviceCount) + " Vulkan-capable device(s)");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        for (const auto& d : devices) {
            if (isDeviceSuitable(d, surface)) {
                physicalDevice = d;
                VkPhysicalDeviceProperties deviceProperties;
                vkGetPhysicalDeviceProperties(d, &deviceProperties);
                logMessage("Selected GPU: " + std::string(deviceProperties.deviceName));
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }

        // Create logical device
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);
        float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        logMessage("Enabling device extensions: " + std::string(VK_KHR_SWAPCHAIN_EXTENSION_NAME));

        if (enableValidationLayers) {
            deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
            logMessage("Enabled validation layers");
        } else {
            deviceCreateInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }
        logMessage("Logical device created successfully");

        // Get device queues
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        logMessage("Device queues retrieved successfully");

        // Create descriptor set layout
        createDescriptorSetLayout();
        
        // Create vertex and index buffers
        createVertexBuffer();
        createIndexBuffer();
        
        // Create uniform buffer
        createUniformBuffer();
        
        // Create descriptor pool and sets
        createDescriptorPool();
        createDescriptorSet();

        // Create swapchain
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        logMessage("Chosen swap chain format: " + std::to_string(surfaceFormat.format));
        logMessage("Chosen present mode: " + std::to_string(presentMode));
        logMessage("Chosen extent: " + std::to_string(extent.width) + "x" + std::to_string(extent.height));

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        logMessage("Using " + std::to_string(imageCount) + " images in swap chain");

        VkSwapchainCreateInfoKHR swapchainCreateInfo{};
        swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainCreateInfo.surface = surface;
        swapchainCreateInfo.minImageCount = imageCount;
        swapchainCreateInfo.imageFormat = globalSurfaceFormat.format;
        swapchainCreateInfo.imageColorSpace = globalSurfaceFormat.colorSpace;
        swapchainCreateInfo.imageExtent = extent;
        swapchainCreateInfo.imageArrayLayers = 1;
        swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // Only use color attachment
        
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapchainCreateInfo.queueFamilyIndexCount = 2;
            swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        swapchainCreateInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchainCreateInfo.presentMode = presentMode;
        swapchainCreateInfo.clipped = VK_TRUE;
        swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

        VkResult result = vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain);
        if (result != VK_SUCCESS) {
            std::string error = "Failed to create swap chain! Error code: " + std::to_string(result);
            logMessage(error, true);
            throw std::runtime_error(error);
        }
        logMessage("Swapchain created successfully");

        // Create render pass
        createRenderPass();

        // Create graphics pipeline
        createGraphicsPipeline();

        // Create image views
        createImageViews();

        // Create framebuffers
        createFramebuffers();

        // Create command pool and buffers
        createCommandPool();
        createCommandBuffer();

        // Create synchronization objects
        createSyncObjects();

        // Main loop
        writeDebugOutput("Entering main loop");

        try {
            while (!glfwWindowShouldClose(window)) {
                try {
                    writeDebugOutput("--- Frame Start ---");
                    glfwPollEvents();
                    writeDebugOutput("Events polled");
                    
                    updateUniformBuffer();
                    writeDebugOutput("Uniform buffer updated");
                    
                    drawFrame();
                    writeDebugOutput("Frame drawn successfully");
                    
                    writeDebugOutput("--- Frame End ---");
                } catch (const std::exception& e) {
                    char buffer[1024];
                    sprintf_s(buffer, "Exception in main loop: %s", e.what());
                    writeDebugOutput(buffer);
                    break;
                }
            }
        } catch (const std::exception& e) {
            char buffer[1024];
            sprintf_s(buffer, "Fatal exception: %s", e.what());
            writeDebugOutput(buffer);
        }

        writeDebugOutput("Main loop ended, starting cleanup");

        // Cleanup
        if (debugFileHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(debugFileHandle);
        }
        
        return 0;
    } catch (const std::exception& e) {
        char buffer[1024];
        sprintf_s(buffer, "Fatal exception: %s", e.what());
        writeDebugOutput(buffer);
        
        if (debugFileHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(debugFileHandle);
        }
        return -1;
    }
}