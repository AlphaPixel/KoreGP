# Two suites (docs/plan.md):
#   koregp-offline   runs on Mesa's nouveau DRM shim; no GPU, no window
#   koregp-hardware  needs the card on nouveau; run through with-nouveau.sh
#
# The same executables serve both. On the shim nothing executes, so the
# offline suite sets KGP_TEST_OFFLINE and the programs skip result checks.

kgp_glslang = find_program('glslangValidator', native : true)

kgp_spv = {}
foreach name : ['saxpy', 'prefix_sum', 'scratch', 'genargs', 'rootloop',
                'heap_copy', 'heap_fetch', 'heap_sample', 'heap_store',
                'bad_push', 'bad_ubo', 'spec', 'multi_gen']
  kgp_spv += {
    name : custom_target(
      name + '_spv.h',
      input : 'shaders/' + name + '.comp',
      output : name + '_spv.h',
      command : [
        kgp_glslang, '--quiet', '--target-env', 'vulkan1.3',
        '--vn', name + '_spv', '-o', '@OUTPUT@', '@INPUT@',
      ],
    ),
  }
endforeach

# The texel-fetch shader again for the other texture types.
foreach name, define : {'heap_fetch_array' : 'FETCH_ARRAY',
                        'heap_fetch_3d' : 'FETCH_3D'}
  kgp_spv += {
    name : custom_target(
      name + '_spv.h',
      input : 'shaders/heap_fetch.comp',
      output : name + '_spv.h',
      command : [
        kgp_glslang, '--quiet', '--target-env', 'vulkan1.3', '-D' + define,
        '--vn', name + '_spv', '-o', '@OUTPUT@', '@INPUT@',
      ],
    ),
  }
endforeach

# Raster shaders: glslang takes the stage from the file extension.
foreach name, file : {'tri_vs' : 'tri.vert', 'tri_ps' : 'tri.frag',
                      'tri_mrt_ps' : 'tri_mrt.frag', 'tex_ps' : 'tex.frag',
                      'bad_ps_push' : 'bad_ps_push.frag'}
  kgp_spv += {
    name : custom_target(
      name + '_spv.h',
      input : 'shaders/' + file,
      output : name + '_spv.h',
      command : [
        kgp_glslang, '--quiet', '--target-env', 'vulkan1.3',
        '--vn', name + '_spv', '-o', '@OUTPUT@', '@INPUT@',
      ],
    ),
  }
endforeach

kgp_offline_env = environment()
kgp_offline_env.set('LD_PRELOAD', libnouveau_noop_drm_shim.full_path())
kgp_offline_env.set('NOUVEAU_CHIPSET', '164')   # TU104
kgp_offline_env.set('KGP_TEST_OFFLINE', '1')

kgp_tests = {
  'kgp_info' : [],
  'kgp_saxpy' : [kgp_spv['saxpy']],
  'kgp_prefix_sum' : [kgp_spv['prefix_sum']],
  'kgp_scratch' : [kgp_spv['scratch']],
  'kgp_indirect' : [kgp_spv['saxpy'], kgp_spv['genargs']],
  'kgp_heap' : [kgp_spv['heap_copy'], kgp_spv['heap_fetch'],
                kgp_spv['heap_sample'], kgp_spv['heap_store']],
  'kgp_copy' : [kgp_spv['heap_fetch'], kgp_spv['heap_fetch_array'],
                kgp_spv['heap_fetch_3d']],
  'kgp_gfx_pipeline' : [kgp_spv['tri_vs'], kgp_spv['tri_ps'],
                        kgp_spv['bad_ps_push'], kgp_spv['saxpy']],
  'kgp_triangle' : [kgp_spv['tri_vs'], kgp_spv['tri_ps'], kgp_spv['saxpy'],
                    kgp_spv['multi_gen'], kgp_spv['tri_mrt_ps']],
  'kgp_textured' : [kgp_spv['tri_vs'], kgp_spv['tex_ps']],
  'kgp_reject' : [kgp_spv['saxpy'], kgp_spv['bad_push'], kgp_spv['bad_ubo']],
  'kgp_spec' : [kgp_spv['spec'], kgp_spv['tri_vs'], kgp_spv['tri_ps']],
  'kgp_present' : [kgp_spv['tri_vs'], kgp_spv['tri_ps']],
  'kgp_validate' : [kgp_spv['saxpy'], kgp_spv['heap_fetch']],
}

kgp_exes = {}
foreach name, generated : kgp_tests
  exe = executable(
    name,
    [name + '.c', generated],
    dependencies : idep_koregp,
    install : true,
  )
  test(name, exe, suite : ['koregp-offline'], env : kgp_offline_env,
       depends : libnouveau_noop_drm_shim)
  test(name, exe, suite : ['koregp-hardware'])
  kgp_exes += {name : exe}
endforeach

# The texture tests again with the untiled layout (the D4 probe needs both).
foreach name : ['kgp_heap', 'kgp_copy']
  test(name + '_linear', kgp_exes[name], args : ['linear'],
       suite : ['koregp-offline'], env : kgp_offline_env,
       depends : libnouveau_noop_drm_shim)
  test(name + '_linear', kgp_exes[name], args : ['linear'],
       suite : ['koregp-hardware'])
endforeach

# The copies again on one mip level, one array layer and a 3D texture.
foreach variant : ['mip', 'layer', '3d']
  test('kgp_copy_' + variant, kgp_exes['kgp_copy'], args : [variant],
       suite : ['koregp-offline'], env : kgp_offline_env,
       depends : libnouveau_noop_drm_shim)
  test('kgp_copy_' + variant, kgp_exes['kgp_copy'], args : [variant],
       suite : ['koregp-hardware'])
endforeach

# Presentation on the monitor: only in a window opened with modesetting on
# (w0.sh present); hardware-suite.sh runs it when a display is connected.
test('kgp_present_display', kgp_exes['kgp_present'], args : ['display'],
     suite : ['koregp-hardware'], is_parallel : false)

# The triangle again with its draw arguments read by the GPU.
test('kgp_triangle_indirect', kgp_exes['kgp_triangle'], args : ['indirect'],
     suite : ['koregp-offline'], env : kgp_offline_env,
     depends : libnouveau_noop_drm_shim)
test('kgp_triangle_indirect', kgp_exes['kgp_triangle'], args : ['indirect'],
     suite : ['koregp-hardware'])

# The triangle again with a depth target and state objects (R7), with a
# depth-stencil target (R7b), as a GPU-driven multi-draw (R9b), and as that
# with a compute shader writing everything the multi-draw reads.
# Last, into render targets of other formats.
foreach variant : ['depth', 'stencil', 'state', 'mrt', 'multi', 'multigpu',
                   'srgb', 'rgba16f', 'rgba32f', 'r32f']
  test('kgp_triangle_' + variant, kgp_exes['kgp_triangle'], args : [variant],
       suite : ['koregp-offline'], env : kgp_offline_env,
       depends : libnouveau_noop_drm_shim)
  test('kgp_triangle_' + variant, kgp_exes['kgp_triangle'], args : [variant],
       suite : ['koregp-hardware'])
endforeach

# The multi-draw macro, simulated (no GPU, no shim).
test('mme_multi_sim', find_program('sh'),
     args : [files('mme_multi_sim.sh'), kgp_mme_sim,
             files('mme_multi_sim.expected')],
     suite : ['koregp-offline'])

# GPU-time benchmark; hardware only, run by hardware-suite.sh.
executable(
  'bench_gpu',
  ['bench_gpu.c', kgp_spv['saxpy'], kgp_spv['rootloop']],
  dependencies : idep_koregp,
  install : true,
)

# The same shaders through Vulkan on the NVK built alongside (the baseline
# for correctness and, later, for dispatch cost). Goes through the system
# Vulkan loader, pointed at this build's ICD only.
dep_vulkan_loader = dependency('vulkan', required : false)
if dep_vulkan_loader.found()
  kgp_vk_env = environment()
  kgp_vk_env.set('VK_DRIVER_FILES', _dev_icd.full_path())

  kgp_vk_offline_env = environment()
  kgp_vk_offline_env.set('VK_DRIVER_FILES', _dev_icd.full_path())
  kgp_vk_offline_env.set('LD_PRELOAD', libnouveau_noop_drm_shim.full_path())
  kgp_vk_offline_env.set('NOUVEAU_CHIPSET', '164')
  kgp_vk_offline_env.set('KGP_TEST_OFFLINE', '1')

  vk_baseline = executable(
    'vk_baseline',
    ['vk_baseline.c', kgp_spv['saxpy'], kgp_spv['prefix_sum'],
     kgp_spv['scratch'], kgp_spv['genargs'], kgp_spv['tri_vs'],
     kgp_spv['tri_ps'], kgp_spv['tri_mrt_ps']],
    dependencies : dep_vulkan_loader,
    include_directories : inc_koregp,
    install : true,
  )
  foreach mode : ['saxpy', 'prefix_sum', 'scratch', 'indirect']
    test('vk_' + mode, vk_baseline, args : [mode],
         suite : ['koregp-offline'], env : kgp_vk_offline_env,
         depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
    test('vk_' + mode, vk_baseline, args : [mode],
         suite : ['koregp-hardware'], env : kgp_vk_env,
         depends : [libvulkan_nouveau, _dev_icd])

    # KoreGP must push exactly what NVK pushes for the same work.
    test('push_matches_nvk_' + mode, find_program('sh'),
         args : [files('push_matches_nvk.sh'), mode,
                 kgp_exes['kgp_' + mode], vk_baseline],
         suite : ['koregp-offline'], env : kgp_vk_offline_env,
         depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
  endforeach

  # First triangle, through Vulkan (the twin of the KoreGP test to come).
  test('vk_triangle', vk_baseline, args : ['triangle'],
       suite : ['koregp-offline'], env : kgp_vk_offline_env,
       depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
  test('vk_triangle', vk_baseline, args : ['triangle'],
       suite : ['koregp-hardware'], env : kgp_vk_env,
       depends : [libvulkan_nouveau, _dev_icd])

  foreach variant : [[], ['indirect'], ['depth'], ['stencil'], ['state'],
                     ['mrt'], ['srgb'], ['rgba16f'], ['rgba32f'], ['r32f']]
    suffix = variant.length() > 0 ? '_' + variant[0] : ''
    test('triangle_matches_nvk' + suffix, find_program('sh'),
         args : [files('triangle_matches_nvk.sh'),
                 files('triangle_matches_nvk.awk'), kgp_exes['kgp_triangle'],
                 vk_baseline] + variant,
         suite : ['koregp-offline'], env : kgp_vk_offline_env,
         depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
    if variant.length() > 0
      test('vk_triangle' + suffix, vk_baseline, args : ['triangle'] + variant,
           suite : ['koregp-offline'], env : kgp_vk_offline_env,
           depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
      test('vk_triangle' + suffix, vk_baseline, args : ['triangle'] + variant,
           suite : ['koregp-hardware'], env : kgp_vk_env,
           depends : [libvulkan_nouveau, _dev_icd])
    endif
  endforeach

  # The 3D engine's channel init must be NVK's.
  test('init3d_matches_nvk', find_program('sh'),
       args : [files('init3d_matches_nvk.sh'), kgp_exes['kgp_info'],
               vk_baseline],
       suite : ['koregp-offline'], env : kgp_vk_offline_env,
       depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])

  # GPU-side copies: the twin records on NVK's transfer-only queue, which
  # the shim has only with mesa-patches/0003.
  foreach layout : ['tiled', 'linear', 'mip', 'layer', '3d']
    layout_args = layout == 'tiled' ? [] : [layout]
    suffix = layout == 'tiled' ? '' : '_' + layout
    test('vk_copy' + suffix, vk_baseline, args : ['copy'] + layout_args,
         suite : ['koregp-offline'], env : kgp_vk_offline_env,
         depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
    test('vk_copy' + suffix, vk_baseline, args : ['copy'] + layout_args,
         suite : ['koregp-hardware'], env : kgp_vk_env,
         depends : [libvulkan_nouveau, _dev_icd])
    test('push_matches_nvk_copy' + suffix, find_program('sh'),
         args : [files('push_matches_nvk.sh'), 'copy',
                 kgp_exes['kgp_copy'], vk_baseline] + layout_args,
         suite : ['koregp-offline'], env : kgp_vk_offline_env,
         depends : [libnouveau_noop_drm_shim, libvulkan_nouveau, _dev_icd])
  endforeach

  # The benchmark itself is run by hand (docs/results-n1.md); it is also the
  # workload for the "a submit binds nothing" check.
  bench_dispatch = executable(
    'bench_dispatch',
    ['bench_dispatch.c', kgp_spv['saxpy']],
    dependencies : [idep_koregp, dep_vulkan_loader],
    install : true,
  )
  test('no_bind_per_submit', find_program('sh'),
       args : [files('no_bind_per_submit.sh'), bench_dispatch],
       suite : ['koregp-offline'], env : kgp_vk_offline_env,
       depends : libnouveau_noop_drm_shim)
endif
