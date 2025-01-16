/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright 2025 Advanced Micro Devices, Inc. or its affiliates. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/utsname.h>
#include "rocm_memory.h"
#include <hip/hip_runtime_api.h>
#if defined HAVE_HIP_HIP_VERSION_H
#include <hip/hip_version.h>
#endif
#include "perftest_parameters.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#define ROCM_CHECK(stmt)			\
	do {					\
	hipError_t result = (stmt);		\
	ASSERT(hipSuccess == result);		\
} while (0)

#define ACCEL_PAGE_SIZE (64 * 1024)


struct rocm_memory_ctx {
	struct memory_ctx base;
	int device_id;
	char *device_bus_id;
	hipDevice_t hipDevice;
	hipCtx_t hipContext;
	bool use_dmabuf;
};


static int init_gpu(struct rocm_memory_ctx *ctx)
{
	int rocm_device_id = ctx->device_id;
	int rocm_pci_bus_id;
	int rocm_pci_device_id;
	int index;
	hipDevice_t hip_device;

	printf("initializing HIP\n");
	hipError_t error = hipInit(0);
	if (error != hipSuccess) {
		printf("hipInit(0) returned %d\n", error);
		return FAILURE;
	}

	int deviceCount = 0;
	error = hipGetDeviceCount(&deviceCount);
	if (error != hipSuccess) {
		printf("hipGetDeviceCount() returned %d\n", error);
		return FAILURE;
	}
	/* This function call returns 0 if there are no HIP capable devices. */
	if (deviceCount == 0) {
		printf("There are no available device(s) that support HIP\n");
		return FAILURE;
	}
	if (rocm_device_id >= deviceCount) {
		fprintf(stderr, "No such device ID (%d) exists in system\n", rocm_device_id);
		return FAILURE;
	}

	printf("Listing all HIP devices in system:\n");
	for (index = 0; index < deviceCount; index++) {
		ROCM_CHECK(hipDeviceGet(&hip_device, index));
		hipDeviceGetAttribute(&rocm_pci_bus_id, hipDeviceAttributePciBusId , hip_device);
		hipDeviceGetAttribute(&rocm_pci_device_id, hipDeviceAttributePciDeviceId , hip_device);
		printf("HIP device %d: PCIe address is %02X:%02X\n", index, (unsigned int)rocm_pci_bus_id, (unsigned int)rocm_pci_device_id);
	}

	printf("\nPicking device No. %d\n", rocm_device_id);

	ROCM_CHECK(hipDeviceGet(&ctx->hipDevice, rocm_device_id));

	char name[128];
	ROCM_CHECK(hipDeviceGetName(name, sizeof(name), rocm_device_id));
	printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), ctx->hipDevice, name);
	printf("creating HIP Ctx\n");

	/* Create context */
	error = hipCtxCreate(&ctx->hipContext, hipDeviceMapHost, ctx->hipDevice);
	if (error != hipSuccess) {
		printf("hipCtxCreate() error=%d\n", error);
		return FAILURE;
	}

	printf("making it the current HIP Ctx\n");
	error = hipCtxSetCurrent(ctx->hipContext);
	if (error != hipSuccess) {
		printf("hipCtxSetCurrent() error=%d\n", error);
		return FAILURE;
	}

	return SUCCESS;
}

static void free_gpu(struct rocm_memory_ctx *ctx)
{
    printf("destroying current HIP Ctx\n");
    ROCM_CHECK(hipCtxDestroy(ctx->hipContext));
}

int rocm_memory_init(struct memory_ctx *ctx) {
	struct rocm_memory_ctx *rocm_ctx = container_of(ctx, struct rocm_memory_ctx, base);
	int return_value = 0;

	if (rocm_ctx->device_bus_id) {
		int err;

		printf("initializing HIP\n");
		hipError_t error = hipInit(0);
		if (error != hipSuccess) {
			printf("hipInit(0) returned %d\n", error);
			return FAILURE;
		}

		printf("Finding PCIe BUS %s\n", rocm_ctx->device_bus_id);
		err = hipDeviceGetByPCIBusId(&rocm_ctx->device_id, rocm_ctx->device_bus_id);
		if (err != 0) {
			fprintf(stderr, "hipDeviceGetByPCIBusId failed with error: %d; Failed to get PCI Bus ID (%s)\n", err, rocm_ctx->device_bus_id);
			return FAILURE;
		}
		printf("Picking GPU number %d\n", rocm_ctx->device_id);
	}

	return_value = init_gpu(rocm_ctx);
	if (return_value) {
		fprintf(stderr, "Couldn't init GPU context: %d\n", return_value);
		return FAILURE;
	}

#ifdef HAVE_ROCM_DMABUF
	if (rocm_ctx->use_dmabuf) {
		int dmabuf_supported = 0;
		const char kernel_opt1[] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
		const char kernel_opt2[] = "CONFIG_PCI_P2PDMA=y";
		int found_opt1           = 0;
		int found_opt2           = 0;
		FILE *fp;
		struct utsname utsname;
		char kernel_conf_file[128];
		char buf[256];

		if (uname(&utsname) == -1) {
			printf("could not get kernel name");
			return FAILURE;
		}

		snprintf(kernel_conf_file, sizeof(kernel_conf_file),
						"/boot/config-%s", utsname.release);
		fp = fopen(kernel_conf_file, "r");
		if (fp == NULL) {
			printf("could not open kernel conf file %s error: %m",
					kernel_conf_file);
			return FAILURE;
		}

		while (fgets(buf, sizeof(buf), fp) != NULL) {
			if (strstr(buf, kernel_opt1) != NULL) {
				found_opt1 = 1;
			}
			if (strstr(buf, kernel_opt2) != NULL) {
				found_opt2 = 1;
			}
			if (found_opt1 && found_opt2) {
				dmabuf_supported = 1;
				break;
			}
		}
		fclose(fp);
	}
#endif

	return SUCCESS;
}

int rocm_memory_destroy(struct memory_ctx *ctx) {
	struct rocm_memory_ctx *rocm_ctx = container_of(ctx, struct rocm_memory_ctx, base);

	free_gpu(rocm_ctx);
	free(rocm_ctx);
	return SUCCESS;
}

int rocm_memory_allocate_buffer(struct memory_ctx *ctx, int alignment, uint64_t size, int *dmabuf_fd,
				uint64_t *dmabuf_offset, void **addr, bool *can_init) {
	hipError_t error;
	size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);

	// Check if discrete or integrated GPU, for allocating memory where adequate
	struct rocm_memory_ctx *rocm_ctx = container_of(ctx, struct rocm_memory_ctx, base);
	int hip_device_integrated;
	hipDeviceGetAttribute(&hip_device_integrated, hipDeviceAttributeIntegrated, rocm_ctx->hipDevice);
	printf("HIP device integrated: %X\n", (unsigned int)hip_device_integrated);

	if (hip_device_integrated == 1) {
		printf("hipHostMalloc() of a %lu bytes GPU buffer\n", size);

		error = hipHostMalloc(addr, buf_size, hipHostMallocDefault);
		if (error != hipSuccess) {
			printf("hipHostMalloc error=%d\n", error);
			return FAILURE;
		}

		printf("allocated GPU buffer address at %p\n", addr);
		*can_init = false;
	} else {
		hipDeviceptr_t d_A;
		printf("hipMalloc() of a %lu bytes GPU buffer\n", size);

		error = hipMalloc(&d_A, buf_size);
		if (error != hipSuccess) {
			printf("hipMalloc error=%d\n", error);
			return FAILURE;
		}

		printf("allocated GPU buffer address at %016llx pointer=%p\n", (unsigned long long)d_A, (void *)d_A);
		*addr = (void *)d_A;
		*can_init = false;

#ifdef HAVE_ROCM_DMABUF
		{
			if (rocm_ctx->use_dmabuf) {
				hipDeviceptr_t aligned_ptr;
				const size_t host_page_size = sysconf(_SC_PAGESIZE);
				uint64_t offset;
				size_t aligned_size;
				hsa_status_t status;

				// Round down to host page size
				aligned_ptr = (hipDeviceptr_t)((uintptr_t)d_A & ~(host_page_size - 1));
				offset = d_A - aligned_ptr;
				aligned_size = (size + offset + host_page_size - 1) & ~(host_page_size - 1);

				printf("using DMA-BUF for GPU buffer address at %#llx aligned at %#llx with aligned size %zu\n", d_A, aligned_ptr, aligned_size);
				*dmabuf_fd = 0;

				status = hsa_amd_portable_export_dmabuf(d_A, aligned_size, dmabuf_fd, &offset);
				if (status != HSA_STATUS_SUCCESS) {
					printf("failed to export dmabuf handle for addr %p / %zu", d_A,
							aligned_size);
					return FAILURE;
				}

				printf("dmabuf export addr %p %lu to dmabuf fd %d offset %zu\n",
						d_A, aligned_size, *dmabuf_fd, offset);

				*dmabuf_offset = offset;
			}
		}
#endif
	}

	return SUCCESS;
}

int rocm_memory_free_buffer(struct memory_ctx *ctx, int dmabuf_fd, void *addr, uint64_t size) {
	struct rocm_memory_ctx *rocm_ctx = container_of(ctx, struct rocm_memory_ctx, base);
	int hip_device_integrated;
	hipDeviceGetAttribute(&hip_device_integrated, hipDeviceAttributeIntegrated, rocm_ctx->hipDevice);

	if (hip_device_integrated == 1) {
		printf("deallocating GPU buffer %p\n", addr);
		hipHostFree(addr);
	} else {
		hipDeviceptr_t d_A = (hipDeviceptr_t)addr;
		printf("deallocating GPU buffer %016llx\n", d_A);
		hipFree(d_A);
	}

	return SUCCESS;
}

void *rocm_memory_copy_host_buffer(void *dest, const void *src, size_t size) {
	hipMemcpy((hipDeviceptr_t)dest, (hipDeviceptr_t)src, size, hipMemcpyHostToDevice);
	return dest;
}

void *rocm_memory_copy_buffer_to_buffer(void *dest, const void *src, size_t size) {
	hipMemcpyDtoD((hipDeviceptr_t)dest, (hipDeviceptr_t)src, size);
	return dest;
}

bool rocm_memory_supported() {
	return true;
}

bool rocm_memory_dmabuf_supported() {
#ifdef HAVE_ROCM_DMABUF
	return true;
#else
	return false;
#endif
}

struct memory_ctx *rocm_memory_create(struct perftest_parameters *params) {
	struct rocm_memory_ctx *ctx;

	ALLOCATE(ctx, struct rocm_memory_ctx, 1);
	ctx->base.init = rocm_memory_init;
	ctx->base.destroy = rocm_memory_destroy;
	ctx->base.allocate_buffer = rocm_memory_allocate_buffer;
	ctx->base.free_buffer = rocm_memory_free_buffer;
	ctx->base.copy_host_to_buffer = rocm_memory_copy_host_buffer;
	ctx->base.copy_buffer_to_host = rocm_memory_copy_host_buffer;
	ctx->base.copy_buffer_to_buffer = rocm_memory_copy_buffer_to_buffer;
	ctx->device_id = params->rocm_device_id;
	ctx->device_bus_id = params->rocm_device_bus_id;
	ctx->use_dmabuf = params->use_rocm_dmabuf;

	return &ctx->base;
}
