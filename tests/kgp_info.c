/* Opens the first KoreGP device and prints what it is. */
#include <koregp/kgp.h>

#include <inttypes.h>
#include <stdio.h>

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   if (dev == NULL)
      return 1;

   KgpDeviceProperties props;
   kgpGetDeviceProperties(dev, &props);
   printf("device        %s\n", props.deviceName);
   printf("pci id        %04x:%04x\n", props.vendorId, props.deviceId);
   printf("chipset       0x%x\n", props.chipset);
   printf("vram          %" PRIu64 " MiB\n", props.vramBytes >> 20);
   printf("mapped vram   %" PRIu64 " MiB\n", props.mappedVramBytes >> 20);

   kgpDestroyDevice(dev);
   return 0;
}
