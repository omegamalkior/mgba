/* 3DS support for the experimental ARM dynarec.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#include "jit-3ds.h"

#include <mgba/internal/arm/arm.h>
#include <mgba/internal/arm/jit.h>

#include <3ds.h>
#include <malloc.h>

/* Written by ctrJitFlushCache, read by the kernel-mode routine */
struct CtrJitFlushRange {
	uint32_t start;
	uint32_t end;
};
struct CtrJitFlushRange ctrJitFlushRange;

/* Cache maintenance, run in kernel mode through svcBackdoor (or Luma3DS's
 * svcCustomBackdoor): clean the data cache over new code, then invalidate the
 * instruction cache and branch predictor so the CPU fetches the new code. */
__asm__(
	"	.arm\n"
	"	.section .text.ctrJitKernelFlush, \"ax\", %progbits\n"
	"	.align 2\n"
	"	.global ctrJitKernelFlush\n"
	"	.type ctrJitKernelFlush, %function\n"
	"ctrJitKernelFlush:\n"
	"	cpsid aif\n"
	"	ldr r2, =ctrJitFlushRange\n"
	"	ldr r0, [r2]\n"
	"	ldr r1, [r2, #4]\n"
	"	bic r0, r0, #31\n"
	"1:\n"
	"	mcr p15, 0, r0, c7, c10, 1\n"
	"	add r0, r0, #32\n"
	"	cmp r0, r1\n"
	"	blo 1b\n"
	"	mov r0, #0\n"
	"	mcr p15, 0, r0, c7, c10, 4\n"
	"	mcr p15, 0, r0, c7, c5, 0\n"
	"	mcr p15, 0, r0, c7, c5, 6\n"
	"	mcr p15, 0, r0, c7, c10, 4\n"
	"	mcr p15, 0, r0, c7, c5, 4\n"
	"	bx lr\n"
	"	.pool\n"
	"	.size ctrJitKernelFlush, . - ctrJitKernelFlush\n"
	"	.section .text.ctrJitCustomBackdoor, \"ax\", %progbits\n"
	"	.align 2\n"
	"	.global ctrJitCustomBackdoor\n"
	"	.type ctrJitCustomBackdoor, %function\n"
	"ctrJitCustomBackdoor:\n"
	"	svc 0x80\n"
	"	bx lr\n"
	"	.size ctrJitCustomBackdoor, . - ctrJitCustomBackdoor\n"
	"	.text\n"
);

/* Luma3DS's svcInvalidateInstructionCacheRange, which Azahar also implements */
__asm__(
	"	.arm\n"
	"	.section .text.ctrJitInvalidateICacheRange, \"ax\", %progbits\n"
	"	.align 2\n"
	"	.global ctrJitInvalidateICacheRange\n"
	"	.type ctrJitInvalidateICacheRange, %function\n"
	"ctrJitInvalidateICacheRange:\n"
	"	svc 0x93\n"
	"	bx lr\n"
	"	.text\n"
);

extern s32 ctrJitInvalidateICacheRange(void* addr, u32 size);
extern s32 ctrJitKernelFlush(void);
extern s32 ctrJitCustomBackdoor(s32 (*func)(void));
extern char __end__[];

static void* _backing = NULL;
static void* _exec = NULL;
static size_t _execSize = 0;
static bool _tried = false;
static bool _rosalina = false;
static bool _emulator = false;
static const char* _status = "Not started";

unsigned ctrJitFlushCount = 0;
size_t ctrJitCacheSize(void) { return _execSize; }
static void _flush(void* start, size_t size) {
	++ctrJitFlushCount;
	if (_emulator) {
		// Emulators translate guest code themselves; just tell them it changed
		ctrJitInvalidateICacheRange(start, size);
		return;
	}
	ctrJitFlushRange.start = (uint32_t) start;
	ctrJitFlushRange.end = (uint32_t) start + size;
	if (_rosalina) {
		ctrJitCustomBackdoor(ctrJitKernelFlush);
	} else {
		svcBackdoor(ctrJitKernelFlush);
	}
}

static bool _initMemory(void) {
	if (_tried) {
		return _exec != NULL;
	}
	_tried = true;

	s64 version = 0;
	_rosalina = R_SUCCEEDED(svcGetSystemInfo(&version, 0x10000, 0));
	s64 emulatorId = 0;
	// Citra/Azahar answer this with 1 or 2; real hardware returns an error
	_emulator = R_SUCCEEDED(svcGetSystemInfo(&emulatorId, 0x20000, 0)) && (emulatorId == 1 || emulatorId == 2);

	// Bigger games (e.g. Metroid Fusion) need more than 4 MiB of translated
	// code; with too little the cache keeps filling up and everything gets
	// recompiled. The New 3DS (and emulators) have the memory to spare.
	bool isNew3DS = false;
	APT_CheckNew3DS(&isNew3DS);
	static const size_t sizes[] = { 0x1000000, 0x800000, 0x400000, 0x200000, 0x100000 };
	size_t i = (isNew3DS || _emulator) ? 0 : 2;
	for (; i < sizeof(sizes) / sizeof(*sizes) && !_backing; ++i) {
		_backing = memalign(0x1000, sizes[i]);
		if (_backing) {
			_execSize = sizes[i];
		}
	}
	if (!_backing) {
		_status = "Not enough memory";
		return false;
	}

	if (_emulator) {
		// Emulators don't enforce execute permission, and don't implement the
		// calls used on hardware, so run the code straight from the buffer
		_exec = _backing;
		_status = "Active (emulator)";
		return true;
	}

	Handle self;
	if (R_FAILED(svcDuplicateHandle(&self, CUR_PROCESS_HANDLE))) {
		_status = "No process handle";
		free(_backing);
		_backing = NULL;
		return false;
	}

	// Map the buffer a second time with execute permission, just below 32 MiB
	// (the same place gpSP uses), as long as that is clear of our own image.
	uint32_t target = 0x02000000 - _execSize;
	Result res = -1;
	if ((uint32_t) __end__ + 0x100000 <= target) {
		res = svcControlProcessMemory(self, target, (u32) _backing, _execSize, MEMOP_MAP, MEMPERM_READ | MEMPERM_WRITE | MEMPERM_EXECUTE);
		if (R_SUCCEEDED(res)) {
			_exec = (void*) target;
		}
	}
	if (R_FAILED(res)) {
		// Otherwise try to make the buffer itself executable
		res = svcControlProcessMemory(self, (u32) _backing, 0, _execSize, MEMOP_PROT, MEMPERM_READ | MEMPERM_WRITE | MEMPERM_EXECUTE);
		if (R_SUCCEEDED(res)) {
			_exec = _backing;
		}
	}
	svcCloseHandle(self);

	if (_exec) {
		// Some environments (e.g. emulators) don't implement the call but don't
		// report an error either, so check that the memory really is executable
		MemInfo info;
		PageInfo page;
		if (R_FAILED(svcQueryMemory(&info, &page, (u32) _exec)) || !(info.perm & MEMPERM_EXECUTE) ||
		    !(info.perm & MEMPERM_WRITE) || info.base_addr > (u32) _exec ||
		    info.base_addr + info.size < (u32) _exec + _execSize) {
			_exec = NULL;
		}
	}
	if (!_exec) {
		_status = "System refused executable memory";
		free(_backing);
		_backing = NULL;
		return false;
	}
	_status = "Active";
	return true;
}

bool ctrJitAttach(struct ARMCore* cpu, bool enabled) {
	if (!enabled && !cpu->jit) {
		return false;
	}
	if (!cpu->jit) {
		if (!_initMemory()) {
			return false;
		}
		if (!ARMJitInit(cpu, _exec, _execSize, _flush)) {
			_status = "Could not start";
			return false;
		}
	}
	if (ARMJitIsEnabled(cpu) != enabled) {
		ARMJitSetEnabled(cpu, enabled);
	}
	return ARMJitIsEnabled(cpu);
}

const char* ctrJitStatus(void) {
	return _status;
}
