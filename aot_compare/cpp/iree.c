// IREE's compiled module behind kernel.h. The module is two generated pieces: module.h, the VM
// program emitted as C, which allocates buffers and records the dispatches; and library.o, the
// dispatches' machine code. The runtime runs them on the single-threaded local-sync device.
#include <stdlib.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/runtime/api.h"
#include "kernel.h"
#include "layout.h"
#include "library.h"
#define EMITC_IMPLEMENTATION
#include "module.h"

// jax.export names the module after the exported function, `jit_npmpc_hess`.
#define FUNCTION "jit_npmpc_hess.main"

#define CHECK_OK(expr)                     \
  do {                                     \
    iree_status_t status_ = (expr);        \
    if (!iree_status_is_ok(status_)) {     \
      iree_status_fprint(stderr, status_); \
      abort();                             \
    }                                      \
  } while (0)

static iree_hal_device_t* g_device;
static iree_runtime_session_t* g_session;
static iree_runtime_call_t g_call;
static iree_hal_buffer_view_t* g_inputs[3];

void kernel_setup(const double* z, const double* p, const double* lam_g) {
  iree_allocator_t host = iree_allocator_system();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  CHECK_OK(iree_runtime_instance_create(&instance_options, host, &instance));

  // The device: dispatches looked up in the linked library, buffers from the heap, and the async
  // I/O pool the 3.12 runtime requires every device to have.
  const iree_hal_executable_library_query_fn_t libraries[] = {jit_npmpc_hess_linked_library_query};
  iree_hal_executable_loader_t* loader = NULL;
  CHECK_OK(iree_hal_static_library_loader_create(1, libraries, iree_hal_executable_import_provider_null(), host, &loader));
  iree_hal_allocator_t* allocator = NULL;
  CHECK_OK(iree_hal_allocator_create_heap(iree_make_cstring_view("local"), host, host, &allocator));
  iree_async_proactor_pool_t* pool = NULL;
  CHECK_OK(iree_async_proactor_pool_create(iree_numa_node_count(), NULL, iree_async_proactor_pool_options_default(), host, &pool));
  iree_hal_device_create_params_t create_params = iree_hal_device_create_params_default();
  create_params.proactor_pool = pool;
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);
  CHECK_OK(iree_hal_sync_device_create(iree_make_cstring_view("local-sync"), &params, &create_params, 1, &loader, allocator, host, &g_device));
  iree_async_proactor_pool_release(pool);
  iree_hal_allocator_release(allocator);
  iree_hal_executable_loader_release(loader);

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  CHECK_OK(iree_runtime_session_create_with_device(instance, &session_options, g_device, host, &g_session));
  iree_vm_module_t* module = NULL;
  CHECK_OK(jit_npmpc_hess_create(iree_runtime_instance_vm_instance(instance), host, &module));
  CHECK_OK(iree_runtime_session_append_module(g_session, module));
  iree_vm_module_release(module);
  iree_runtime_instance_release(instance);
  CHECK_OK(iree_runtime_call_initialize_by_name(g_session, iree_make_cstring_view(FUNCTION), &g_call));

  const double* data[3] = {z, p, lam_g};
  const iree_hal_dim_t sizes[3] = {NZ, NP, NLAM};
  iree_hal_buffer_params_t buffer_params = {.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT};
  for (int i = 0; i < 3; ++i) {
    CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(g_device, iree_hal_device_allocator(g_device), 1, &sizes[i], IREE_HAL_ELEMENT_TYPE_FLOAT_64,
                                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
                                                       iree_make_const_byte_span(data[i], sizes[i] * sizeof(double)), &g_inputs[i]));
  }
}

// The runtime allocates the result for each call, so it is copied to caller memory and released.
void kernel_evaluate(double* out) {
  iree_runtime_call_reset(&g_call);
  for (int i = 0; i < 3; ++i) CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&g_call, g_inputs[i]));
  CHECK_OK(iree_runtime_call_invoke(&g_call, 0));
  iree_hal_buffer_view_t* result = NULL;
  CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&g_call, &result));
  CHECK_OK(iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(result), 0, out, NNZ * sizeof(double)));
  iree_hal_buffer_view_release(result);
}
