/* The least Vulkan that runs a compute shader taking one push-constant
 * pointer: the baseline the KoreGP tests are compared against. Same SPIR-V,
 * same driver (NVK), through the Vulkan entry points.
 */
#ifndef VK_COMPUTE_H
#define VK_COMPUTE_H

#include "kgp_test.h"

#include <vulkan/vulkan.h>

#define VK_OK(expr) CHECK((expr) == VK_SUCCESS)

struct vkc {
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice dev;
   VkQueue queue;
   VkCommandPool pool;
   VkPipelineLayout layout;   /* one 8-byte push constant: the root pointer */
   uint32_t host_mem_type;
};

struct vkc_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
   VkDeviceAddress addr;
};

/* `transfer_only`: use NVK's transfer queue family instead of the one with
 * compute. That queue is the only place NVK drives the copy engine directly.
 */
static inline void
vkc_init(struct vkc *c, bool transfer_only)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   VK_OK(vkCreateInstance(&(VkInstanceCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   }, NULL, &c->instance));

   uint32_t count = 1;
   VkResult result = vkEnumeratePhysicalDevices(c->instance, &count, &c->pdev);
   CHECK((result == VK_SUCCESS || result == VK_INCOMPLETE) && count == 1);

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(c->pdev, &props);
   fprintf(stderr, "vulkan device: %s\n", props.deviceName);

   VkQueueFamilyProperties families[16];
   uint32_t family_count = 16;
   vkGetPhysicalDeviceQueueFamilyProperties(c->pdev, &family_count, families);
   const VkQueueFlags want_flags =
      transfer_only ? VK_QUEUE_TRANSFER_BIT : VK_QUEUE_COMPUTE_BIT;
   const VkQueueFlags reject_flags =
      transfer_only ? VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT : 0;
   uint32_t family = 0;
   while (family < family_count &&
          (!(families[family].queueFlags & want_flags) ||
           (families[family].queueFlags & reject_flags)))
      family++;
   if (family == family_count) {
      fprintf(stderr, "no %s queue family\n",
              transfer_only ? "transfer-only" : "compute");
      exit(1);
   }

   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE,
   };
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
      .bufferDeviceAddress = VK_TRUE,
      .scalarBlockLayout = VK_TRUE,
   };
   VK_OK(vkCreateDevice(c->pdev, &(VkDeviceCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = family,
         .queueCount = 1,
         .pQueuePriorities = &(float) { 1.0f },
      },
   }, NULL, &c->dev));
   vkGetDeviceQueue(c->dev, family, 0, &c->queue);

   VK_OK(vkCreateCommandPool(c->dev, &(VkCommandPoolCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = family,
   }, NULL, &c->pool));

   VK_OK(vkCreatePipelineLayout(c->dev, &(VkPipelineLayoutCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange) {
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .size = sizeof(VkDeviceAddress),
      },
   }, NULL, &c->layout));

   VkPhysicalDeviceMemoryProperties mem;
   vkGetPhysicalDeviceMemoryProperties(c->pdev, &mem);
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   c->host_mem_type = 0;
   while (c->host_mem_type < mem.memoryTypeCount &&
          (mem.memoryTypes[c->host_mem_type].propertyFlags & want) != want)
      c->host_mem_type++;
   CHECK(c->host_mem_type < mem.memoryTypeCount);
}

static inline void
vkc_finish(struct vkc *c)
{
   vkDestroyPipelineLayout(c->dev, c->layout, NULL);
   vkDestroyCommandPool(c->dev, c->pool, NULL);
   vkDestroyDevice(c->dev, NULL);
   vkDestroyInstance(c->instance, NULL);
}

static inline struct vkc_buffer
vkc_buffer_create(struct vkc *c, VkDeviceSize size)
{
   struct vkc_buffer b;

   VK_OK(vkCreateBuffer(c->dev, &(VkBufferCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
   }, NULL, &b.buffer));

   VkMemoryRequirements reqs;
   vkGetBufferMemoryRequirements(c->dev, b.buffer, &reqs);
   CHECK(reqs.memoryTypeBits & (1u << c->host_mem_type));

   VK_OK(vkAllocateMemory(c->dev, &(VkMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &(VkMemoryAllocateFlagsInfo) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
         .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
      },
      .allocationSize = reqs.size,
      .memoryTypeIndex = c->host_mem_type,
   }, NULL, &b.memory));
   VK_OK(vkBindBufferMemory(c->dev, b.buffer, b.memory, 0));
   VK_OK(vkMapMemory(c->dev, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map));

   b.addr = vkGetBufferDeviceAddress(c->dev, &(VkBufferDeviceAddressInfo) {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = b.buffer,
   });
   CHECK(b.addr);
   return b;
}

static inline void
vkc_buffer_destroy(struct vkc *c, struct vkc_buffer *b)
{
   vkDestroyBuffer(c->dev, b->buffer, NULL);
   vkFreeMemory(c->dev, b->memory, NULL);
}

static inline VkPipeline
vkc_pipeline_create(struct vkc *c, const uint32_t *spirv, size_t size)
{
   VkShaderModule module;
   VK_OK(vkCreateShaderModule(c->dev, &(VkShaderModuleCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = spirv,
   }, NULL, &module));

   VkPipeline pipeline;
   VK_OK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1,
      &(VkComputePipelineCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main",
         },
         .layout = c->layout,
      }, NULL, &pipeline));

   vkDestroyShaderModule(c->dev, module, NULL);
   return pipeline;
}

static inline VkCommandBuffer
vkc_cmd_alloc(struct vkc *c)
{
   VkCommandBuffer cmd;
   VK_OK(vkAllocateCommandBuffers(c->dev, &(VkCommandBufferAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = c->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   }, &cmd));
   return cmd;
}

static inline void
vkc_cmd_begin(VkCommandBuffer cmd)
{
   VK_OK(vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo) {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   }));
}

/* The Vulkan spelling of kgpDispatch(cb, root, groups). */
static inline void
vkc_cmd_dispatch(struct vkc *c, VkCommandBuffer cmd, VkDeviceAddress root,
                 uint32_t x, uint32_t y, uint32_t z)
{
   vkCmdPushConstants(cmd, c->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(root), &root);
   vkCmdDispatch(cmd, x, y, z);
}

static inline void
vkc_cmd_submit(struct vkc *c, VkCommandBuffer cmd)
{
   VK_OK(vkEndCommandBuffer(cmd));
   VK_OK(vkQueueSubmit(c->queue, 1, &(VkSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   }, VK_NULL_HANDLE));
}

#endif
