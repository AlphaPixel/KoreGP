/*
 * Presentation (proposal §5.6): a hidden NVK device that exists only to own
 * a VkSwapchainKHR. The swapchain's images are shared into KoreGP through
 * their BOs (dma-buf) and wrapped as KgpTextures. This is the only file
 * that touches Vulkan.
 *
 * There is no Vulkan loader in between: NVK is linked into this library,
 * and its entry points are reached through Mesa's dispatch tables.
 *
 * ponytail: two kinds of surface, the first connected display at its first
 * mode (VK_KHR_display, so no compositor) and a headless one for tests;
 * Wayland and X11 when a desktop needs them. Presenting does not wait for
 * KoreGP's rendering, the caller does; a sync object handed across to the
 * hidden device when that wait costs a frame.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "nouveau/vulkan/nvk_device_memory.h"
#include "nouveau/vulkan/nvk_image.h"
#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "wsi_common_private.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#define KGP_MAX_SWAPCHAIN_IMAGES 8

/* nvk_entrypoints.h is generated into NVK's build directory. */
VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkInstance *pInstance);

struct KgpSwapchain_T {
   struct KgpDevice_T *dev;
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice device;
   VkQueue queue;
   VkSurfaceKHR surface;
   VkSwapchainKHR swapchain;
   VkFence fence;
   uint32_t image_count;
   struct KgpTexture_T *textures[KGP_MAX_SWAPCHAIN_IMAGES];
   uint32_t acquired;   /* image index; UINT32_MAX: none */
};

#define INST(sc) (&vk_instance_from_handle((sc)->instance)->dispatch_table)
#define PDEV(sc) (&vk_physical_device_from_handle((sc)->pdev)->dispatch_table)
#define DEV(sc)  (&vk_device_from_handle((sc)->device)->dispatch_table)

KGP_API void
kgpDestroySwapchain(KgpSwapchain sc)
{
   if (sc == NULL)
      return;
   for (uint32_t i = 0; i < sc->image_count; i++)
      kgpDestroyTexture(sc->dev, sc->textures[i]);
   if (sc->device != VK_NULL_HANDLE) {
      DEV(sc)->DestroyFence(sc->device, sc->fence, NULL);
      DEV(sc)->DestroySwapchainKHR(sc->device, sc->swapchain, NULL);
   }
   if (sc->instance != VK_NULL_HANDLE)
      INST(sc)->DestroySurfaceKHR(sc->instance, sc->surface, NULL);
   if (sc->device != VK_NULL_HANDLE)
      DEV(sc)->DestroyDevice(sc->device, NULL);
   if (sc->instance != VK_NULL_HANDLE)
      INST(sc)->DestroyInstance(sc->instance, NULL);
   free(sc);
}

/* Why no display was listed. NVK's WSI opens the DRM primary node when the
 * instance is created and keeps it only if that made it DRM master
 * (wsi_display_init_wsi), and it drives the display through atomic
 * modesetting only, dropping a connector whose atomic properties it cannot
 * see (find_properties). In each case it lists nothing and says nothing.
 * The open is repeated here to name the reason.
 */
static void
kgp_swapchain_no_display(struct KgpSwapchain_T *sc)
{
   drmDevicePtr drm = NULL;
   if (drmGetDeviceFromDevId(sc->dev->pdev->drm.primary_dev, 0, &drm) != 0) {
      kgp_err("kgpCreateSwapchain: the GPU has no DRM primary node");
      return;
   }
   const char *node = drm->nodes[DRM_NODE_PRIMARY];
   const int fd = open(node, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      kgp_err("kgpCreateSwapchain: cannot open %s: %s", node,
              strerror(errno));
   } else if (drmAuthMagic(fd, 0) == -EACCES) {
      /* wsi_common_display.c local_drmIsMaster() */
      kgp_err("kgpCreateSwapchain: another process is DRM master of %s (a "
              "compositor or display manager)", node);
   } else if (drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
      kgp_err("kgpCreateSwapchain: the kernel driver of %s does not expose "
              "atomic modesetting (nouveau: load it with atomic=1)", node);
   } else {
      kgp_err("kgpCreateSwapchain: no display is connected to %s", node);
   }
   if (fd >= 0)
      close(fd);
   drmFreeDevice(&drm);
}

/* The surface, and the size its swapchain must have. */
static bool
kgp_swapchain_surface(struct KgpSwapchain_T *sc, const KgpSwapchainDesc *desc,
                      VkExtent2D *extent)
{
   if (desc->target == KGP_PRESENT_HEADLESS) {
      const VkHeadlessSurfaceCreateInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
      };
      *extent = (VkExtent2D) { desc->width, desc->height };
      return desc->width > 0 && desc->height > 0 &&
             INST(sc)->CreateHeadlessSurfaceEXT(sc->instance, &info, NULL,
                                                &sc->surface) == VK_SUCCESS;
   }

   /* Only connected displays are listed. */
   uint32_t n = 1;
   VkDisplayPropertiesKHR display;
   VkResult r = PDEV(sc)->GetPhysicalDeviceDisplayPropertiesKHR(sc->pdev, &n,
                                                                &display);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0) {
      kgp_swapchain_no_display(sc);
      return false;
   }
   /* The kernel lists a connector's preferred mode first. */
   n = 1;
   VkDisplayModePropertiesKHR mode;
   r = PDEV(sc)->GetDisplayModePropertiesKHR(sc->pdev, display.display, &n,
                                             &mode);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0) {
      kgp_err("kgpCreateSwapchain: display \"%s\" has no modes",
              display.displayName);
      return false;
   }
   const VkDisplaySurfaceCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
      .displayMode = mode.displayMode,
      .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .globalAlpha = 1.0f,
      .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
      .imageExtent = mode.parameters.visibleRegion,
   };
   *extent = mode.parameters.visibleRegion;
   return INST(sc)->CreateDisplayPlaneSurfaceKHR(sc->instance, &info, NULL,
                                                 &sc->surface) == VK_SUCCESS;
}

KGP_API KgpSwapchain
kgpCreateSwapchain(KgpDevice dev, const KgpSwapchainDesc *desc,
                   KgpSwapchainInfo *info_out)
{
   if (desc == NULL || info_out == NULL ||
       desc->target > KGP_PRESENT_HEADLESS) {
      kgp_err("kgpCreateSwapchain: bad arguments");
      return NULL;
   }
   const bool headless = desc->target == KGP_PRESENT_HEADLESS;
   const char *step = "out of memory";
   VkResult r = VK_SUCCESS;

   struct KgpSwapchain_T *sc = calloc(1, sizeof(*sc));
   if (sc == NULL)
      goto fail;
   sc->dev = dev;
   sc->acquired = UINT32_MAX;

   step = "hidden Vulkan instance";
   const char *instance_exts[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      headless ? VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME
               : VK_KHR_DISPLAY_EXTENSION_NAME,
   };
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = instance_exts,
   };
   r = nvk_CreateInstance(&instance_info, NULL, &sc->instance);
   if (r != VK_SUCCESS) {
      sc->instance = VK_NULL_HANDLE;
      goto fail;
   }

   step = "hidden Vulkan device";
   uint32_t n = 1;
   r = INST(sc)->EnumeratePhysicalDevices(sc->instance, &n, &sc->pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0)
      goto fail;
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_exts,
   };
   r = PDEV(sc)->CreateDevice(sc->pdev, &device_info, NULL, &sc->device);
   if (r != VK_SUCCESS) {
      sc->device = VK_NULL_HANDLE;
      goto fail;
   }
   DEV(sc)->GetDeviceQueue(sc->device, 0, 0, &sc->queue);

   step = "surface";
   VkExtent2D extent;
   r = VK_SUCCESS;
   if (!kgp_swapchain_surface(sc, desc, &extent))
      goto fail;

   step = "swapchain";
   VkSurfaceCapabilitiesKHR caps;
   r = PDEV(sc)->GetPhysicalDeviceSurfaceCapabilitiesKHR(sc->pdev,
                                                         sc->surface, &caps);
   if (r != VK_SUCCESS)
      goto fail;
   const VkSwapchainCreateInfoKHR swapchain_info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = sc->surface,
      .minImageCount = MAX2(caps.minImageCount, 2),
      /* The one format both kinds of surface offer. */
      .imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
      .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha =
         (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ?
         VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE,
   };
   r = DEV(sc)->CreateSwapchainKHR(sc->device, &swapchain_info, NULL,
                                   &sc->swapchain);
   if (r != VK_SUCCESS)
      goto fail;
   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   r = DEV(sc)->CreateFence(sc->device, &fence_info, NULL, &sc->fence);
   if (r != VK_SUCCESS)
      goto fail;

   step = "sharing the swapchain images";
   VkImage images[KGP_MAX_SWAPCHAIN_IMAGES];
   n = KGP_MAX_SWAPCHAIN_IMAGES;
   r = DEV(sc)->GetSwapchainImagesKHR(sc->device, sc->swapchain, &n, images);
   if (r != VK_SUCCESS)
      goto fail;
   VK_FROM_HANDLE(wsi_swapchain, chain, sc->swapchain);
   for (uint32_t i = 0; i < n; i++) {
      /* "The handle crosses cleanly": the image's layout is NIL's, on
       * both sides, and its memory is a BO either device can import.
       */
      VK_FROM_HANDLE(nvk_image, image, images[i]);
      VK_FROM_HANDLE(nvk_device_memory, memory,
                     chain->get_wsi_image(chain, i)->memory);
      if (memory == NULL || !(memory->mem->flags & NVKMD_MEM_SHARED)) {
         r = VK_ERROR_FEATURE_NOT_PRESENT;
         goto fail;
      }
      int fd;
      r = nvkmd_mem_export_dma_buf(memory->mem, KGP_LOG_OBJ, &fd);
      if (r != VK_SUCCESS)
         goto fail;
      struct nvkmd_mem *imported;
      r = nvkmd_dev_import_dma_buf(dev->nvkmd, KGP_LOG_OBJ, fd, &imported);
      close(fd);
      if (r != VK_SUCCESS)
         goto fail;
      sc->textures[i] = kgp_texture_over_mem(dev, &image->planes[0].nil,
                                             KGP_FORMAT_BGRA8_UNORM, imported,
                                             image->planes[0].plane_offset_B);
      if (sc->textures[i] == NULL) {
         r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto fail;
      }
      sc->image_count = i + 1;
   }

   *info_out = (KgpSwapchainInfo) {
      .format = KGP_FORMAT_BGRA8_UNORM,
      .width = extent.width,
      .height = extent.height,
   };
   return sc;

fail:
   kgp_err("kgpCreateSwapchain: %s failed (VkResult %d)", step, r);
   kgpDestroySwapchain(sc);
   return NULL;
}

KGP_API KgpTexture
kgpAcquireSwapchainTexture(KgpSwapchain sc)
{
   if (sc->acquired != UINT32_MAX) {
      kgp_err("kgpAcquireSwapchainTexture: one is acquired and not yet "
              "presented");
      return NULL;
   }
   uint32_t index;
   VkResult r = DEV(sc)->AcquireNextImageKHR(sc->device, sc->swapchain,
                                             UINT64_MAX, VK_NULL_HANDLE,
                                             sc->fence, &index);
   if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
      kgp_err("kgpAcquireSwapchainTexture: failed (VkResult %d)", r);
      return NULL;
   }
   DEV(sc)->WaitForFences(sc->device, 1, &sc->fence, VK_TRUE, UINT64_MAX);
   DEV(sc)->ResetFences(sc->device, 1, &sc->fence);
   sc->acquired = index;
   return sc->textures[index];
}

KGP_API bool
kgpPresent(KgpSwapchain sc)
{
   if (sc->acquired == UINT32_MAX) {
      kgp_err("kgpPresent: no texture is acquired");
      return false;
   }
   const VkPresentInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .swapchainCount = 1,
      .pSwapchains = &sc->swapchain,
      .pImageIndices = &sc->acquired,
   };
   VkResult r = DEV(sc)->QueuePresentKHR(sc->queue, &info);
   sc->acquired = UINT32_MAX;
   if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
      kgp_err("kgpPresent: failed (VkResult %d)", r);
      return false;
   }
   return true;
}
