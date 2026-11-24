/* Shared by the KoreGP test programs. */
#ifndef KGP_TEST_H
#define KGP_TEST_H

#include <koregp/kgp.h>

#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond) do {                                             \
   if (!(cond)) {                                                    \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n",                   \
              __FILE__, __LINE__, #cond);                            \
      exit(1);                                                       \
   }                                                                 \
} while (0)

/* On the DRM shim nothing executes, so results cannot be checked; the
 * offline suite sets this (tests/meson.build).
 */
static inline bool
kgp_test_offline(void)
{
   return getenv("KGP_TEST_OFFLINE") != NULL;
}

#define KGP_TEST_TIMEOUT_NS (10ull * 1000 * 1000 * 1000)

#endif
