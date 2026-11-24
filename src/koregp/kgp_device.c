/*
 * Device open/close and properties.
 *
 * Ancestors: nvk_physical_device.c:nvk_create_drm_physical_device and
 * nvk_device.c:nvk_CreateDevice, with the vk_instance / vk_physical_device /
 * vk_device objects cut. What is left is the nvkmd calls.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "clc5c0.h"
#include "compiler/glsl_types.h"
#include "util/u_debug.h"

#include <stdlib.h>
#include <xf86drm.h>

/* KGP_DEBUG=push,sync,...  The first four are nvkmd's own debug paths. */
static const struct debug_control kgp_nvkmd_debug_options[] = {
   { "push", NVK_DEBUG_PUSH_DUMP },   /* dump every submitted pushbuf */
   { "sync", NVK_DEBUG_PUSH_SYNC },   /* wait after every submit; dump on error */
   { "vm",   NVK_DEBUG_VM },          /* log VA binds */
   { "gart", NVK_DEBUG_FORCE_GART },  /* all memory in system RAM */
   { NULL, 0 },
};

static const struct debug_control kgp_debug_options[] = {
   { "asm", KGP_DEBUG_ASM },
   { "nir", KGP_DEBUG_NIR },
   { "1", KGP_DEBUG_VALIDATE },       /* the proposal's spelling */
   { "validate", KGP_DEBUG_VALIDATE },
   { NULL, 0 },
};

void
kgp_device_poll_locked(struct KgpDevice_T *dev)
{
   if (dev->done_seq == dev->submit_seq)
      return;

   if (dev->no_fences || dev->lost) {
      dev->done_seq = dev->submit_seq;
   } else {
      uint64_t point = 0;
      if (drmSyncobjQuery(dev->drm_fd, &dev->timeline, &point, 1) == 0 &&
          point > dev->done_seq)
         dev->done_seq = point;
   }

   kgp_queue_reap_locked(&dev->queue);
   kgp_memory_reap_locked(dev);
}

KGP_API KgpDevice
kgpCreateDevice(uint32_t index)
{
   const char *debug_str = os_get_option("KGP_DEBUG");

   drmDevicePtr drm_devs[16];
   const int drm_count = drmGetDevices2(0, drm_devs, ARRAY_SIZE(drm_devs));

   struct nvkmd_pdev *pdev = NULL;
   for (int i = 0; i < drm_count && pdev == NULL; i++) {
      struct nvkmd_pdev *candidate;
      VkResult result = nvkmd_try_create_pdev_for_drm(
         drm_devs[i], KGP_LOG_OBJ,
         parse_debug_string(debug_str, kgp_nvkmd_debug_options), &candidate);
      if (result != VK_SUCCESS)
         continue;

      if (index-- == 0)
         pdev = candidate;
      else
         nvkmd_pdev_destroy(candidate);
   }
   if (drm_count > 0)
      drmFreeDevices(drm_devs, drm_count);

   if (pdev == NULL) {
      kgp_err("no NVIDIA GPU on the nouveau kernel driver at that index");
      return NULL;
   }

   /* Absolute shader addresses in the QMD (Volta+) and 64-byte cbuf
    * alignment (Turing+) are assumed throughout. Older chips need a shader
    * heap with a program region base; not written.
    */
   if (pdev->dev_info.cls_compute < TURING_COMPUTE_A) {
      kgp_err("%s: compute class 0x%04x is older than Turing; unsupported",
              pdev->dev_info.device_name, pdev->dev_info.cls_compute);
      goto fail_pdev;
   }

   struct KgpDevice_T *dev = calloc(1, sizeof(*dev));
   if (dev == NULL)
      goto fail_pdev;

   dev->pdev = pdev;
   dev->debug = parse_debug_string(debug_str, kgp_debug_options);
   dev->no_fences = debug_get_bool_option("NOUVEAU_DISABLE_FENCES", false);
   simple_mtx_init(&dev->mutex, mtx_plain);
   rb_tree_init(&dev->allocs[0]);
   rb_tree_init(&dev->allocs[1]);
   list_inithead(&dev->pending_free);
   dev->old_slm_mems = UTIL_DYNARRAY_INIT;

   if (nvkmd_pdev_create_dev(pdev, KGP_LOG_OBJ, &dev->nvkmd) != VK_SUCCESS)
      goto fail_dev;

   dev->drm_fd = nvkmd_dev_get_drm_fd(dev->nvkmd);
   if (drmSyncobjCreate(dev->drm_fd, 0, &dev->timeline) != 0) {
      kgp_err("drmSyncobjCreate failed: %m");
      goto fail_nvkmd;
   }

   /* spirv_to_nir builds GLSL types; the type cache is refcounted. */
   glsl_type_singleton_init_or_ref();
   dev->nak = nak_compiler_create(&pdev->dev_info);
   if (dev->nak == NULL)
      goto fail_timeline;

   if (!kgp_queue_init(dev))
      goto fail_nak;

   return dev;

fail_nak:
   nak_compiler_destroy(dev->nak);
fail_timeline:
   glsl_type_singleton_decref();
   drmSyncobjDestroy(dev->drm_fd, dev->timeline);
fail_nvkmd:
   nvkmd_dev_destroy(dev->nvkmd);
fail_dev:
   simple_mtx_destroy(&dev->mutex);
   free(dev);
fail_pdev:
   nvkmd_pdev_destroy(pdev);
   return NULL;
}

KGP_API void
kgpDestroyDevice(KgpDevice dev)
{
   if (dev == NULL)
      return;

   /* Wait for the channel to drain; then everything submitted is done. */
   VkResult result = nvkmd_ctx_sync(dev->queue.ctx, KGP_LOG_OBJ);
   if (result != VK_SUCCESS)
      kgp_err("channel did not idle cleanly at device destruction");

   simple_mtx_lock(&dev->mutex);
   dev->done_seq = dev->submit_seq;
   kgp_queue_reap_locked(&dev->queue);
   kgp_memory_reap_locked(dev);
   simple_mtx_unlock(&dev->mutex);

   kgp_queue_finish(dev);
   kgp_memory_finish(dev);

   nak_compiler_destroy(dev->nak);
   glsl_type_singleton_decref();
   drmSyncobjDestroy(dev->drm_fd, dev->timeline);
   nvkmd_dev_destroy(dev->nvkmd);
   nvkmd_pdev_destroy(dev->pdev);
   simple_mtx_destroy(&dev->mutex);
   free(dev);
}

KGP_API void
kgpGetDeviceProperties(KgpDevice dev, KgpDeviceProperties *out)
{
   const struct nv_device_info *info = kgp_dev_info(dev);

   memset(out, 0, sizeof(*out));
   snprintf(out->deviceName, sizeof(out->deviceName), "%s", info->device_name);
   out->vendorId = NVIDIA_VENDOR_ID;
   out->deviceId = info->device_id;
   out->chipset = info->chipset;
   out->vramBytes = info->vram_size_B;
   /* Delta D1. bar_size_B is 0 when the kernel does not report it. */
   out->mappedVramBytes = info->bar_size_B ?
      MIN2(info->bar_size_B, info->vram_size_B) : info->vram_size_B;
}

KGP_API KgpQueue
kgpGetQueue(KgpDevice dev)
{
   return &dev->queue;
}
