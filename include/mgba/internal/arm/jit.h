/* Experimental ARM-on-ARM dynamic recompiler for mGBA.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef ARM_JIT_H
#define ARM_JIT_H

#include <mgba-util/common.h>

CXX_GUARD_START

struct ARMCore;

#define ARM_JIT_PAGE_SHIFT 6
#define ARM_JIT_EWRAM_SIZE 0x40000
#define ARM_JIT_IWRAM_SIZE 0x8000
#define ARM_JIT_RAM_PAGES ((ARM_JIT_EWRAM_SIZE + ARM_JIT_IWRAM_SIZE) >> ARM_JIT_PAGE_SHIFT)

struct ARMJitStats {
	uint64_t blocksCompiled;
	uint64_t blocksEntered;
	uint64_t interpretedSteps;
	uint64_t invalidations;
	uint64_t flushes;
	uint64_t compileFailures;
	uint64_t prefetchMismatches;
	uint32_t lastFailAddress;
	uint32_t hotBytes; // translated code in use: block bodies
	uint32_t coldBytes; // and their out-of-line parts
};

/* The JIT is only usable when mGBA itself runs on a 32-bit ARM host. */
#if defined(__arm__) && !defined(__thumb2__) && !defined(DISABLE_ARM_JIT)
#define ARM_JIT_SUPPORTED 1
#else
#define ARM_JIT_SUPPORTED 0
#endif

typedef void (*ARMJitCacheFlush)(void* start, size_t size);

/* execMemory must be readable, writable and executable. flush is called after
 * new code is written; pass NULL to use the compiler's __builtin___clear_cache. */
bool ARMJitInit(struct ARMCore* cpu, void* execMemory, size_t size, ARMJitCacheFlush flush);
void ARMJitDeinit(struct ARMCore* cpu);
void ARMJitSetEnabled(struct ARMCore* cpu, bool enabled);
bool ARMJitIsEnabled(struct ARMCore* cpu);
void ARMJitFlush(struct ARMCore* cpu);
void ARMJitInvalidateWrite(struct ARMCore* cpu, unsigned offset);
void ARMJitRunLoop(struct ARMCore* cpu);
const struct ARMJitStats* ARMJitGetStats(struct ARMCore* cpu);

/* Called by memory code whenever guest RAM is written. OFFSET is the byte offset
 * into the combined EWRAM+IWRAM buffer (IWRAM starts at ARM_JIT_EWRAM_SIZE). */
#define ARM_JIT_NOTE_WRITE(CPU, OFFSET) \
	do { \
		if (UNLIKELY((CPU)->jitCodePages != NULL)) { \
			unsigned _jitPage = (unsigned) (OFFSET) >> ARM_JIT_PAGE_SHIFT; \
			if (UNLIKELY((CPU)->jitCodePages[_jitPage])) { \
				ARMJitInvalidateWrite((CPU), (unsigned) (OFFSET)); \
			} \
		} \
	} while (0)

#define ARM_JIT_NOTE_EWRAM_WRITE(CPU, ADDRESS) ARM_JIT_NOTE_WRITE(CPU, (ADDRESS) & (ARM_JIT_EWRAM_SIZE - 1))
#define ARM_JIT_NOTE_IWRAM_WRITE(CPU, ADDRESS) ARM_JIT_NOTE_WRITE(CPU, ARM_JIT_EWRAM_SIZE + ((ADDRESS) & (ARM_JIT_IWRAM_SIZE - 1)))

#define ARM_JIT_FLUSH_IF_ACTIVE(CPU) \
	do { \
		if ((CPU)->jit) { \
			ARMJitFlush(CPU); \
		} \
	} while (0)

CXX_GUARD_END

#endif
