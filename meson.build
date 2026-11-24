# KoreGP frontend. This directory is symlinked into the Mesa tree as
# src/koregp by scripts/host/build-mesa.sh and built by Mesa's Meson with
# -Dkoregp=true (mesa-patches/0001). It is not a standalone project.
#
# libnvk is a static library: only the objects KoreGP references end up in
# libkoregp.so. kgp_present.c references NVK's instance, which is all of it.

inc_koregp = include_directories('include')

libkoregp = shared_library(
  'koregp',
  files(
    'src/koregp/kgp_barrier.c',
    'src/koregp/kgp_cmd.c',
    'src/koregp/kgp_copy.c',
    'src/koregp/kgp_device.c',
    'src/koregp/kgp_draw.c',
    'src/koregp/kgp_memory.c',
    'src/koregp/kgp_mme.c',
    'src/koregp/kgp_nir_lower_heap.c',
    'src/koregp/kgp_nir_lower_root.c',
    'src/koregp/kgp_pipeline.c',
    'src/koregp/kgp_present.c',
    'src/koregp/kgp_sampler.c',
    'src/koregp/kgp_sync.c',
    'src/koregp/kgp_texture.c',
    'src/koregp/kgp_validate.c',
  ),
  include_directories : [inc_koregp, inc_include, inc_src],
  dependencies : [nvk_deps, idep_vtn],
  link_with : [libnvk],
  link_args : [ld_args_build_id, ld_args_bsymbolic, ld_args_gc_sections],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

install_headers('include/koregp/kgp.h', subdir : 'koregp')

idep_koregp = declare_dependency(
  link_with : libkoregp,
  include_directories : inc_koregp,
)

# KoreGP's own macro on Mesa's macro-processor simulator (tests/meson.build
# runs it).
kgp_mme_sim = executable(
  'kgp_mme_sim',
  files('src/koregp/kgp_mme.c', 'src/koregp/kgp_mme_sim.c'),
  include_directories : [inc_koregp, inc_include, inc_src],
  dependencies : [nvk_deps, idep_vtn],
  c_args : [no_override_init_args],
)

subdir('tests')
