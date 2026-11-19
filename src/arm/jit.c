/* Experimental ARM-on-ARM dynamic recompiler for mGBA.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Design overview
 * ---------------
 * Guest code (GBA ARM7TDMI, ARM and Thumb) is translated in blocks into host
 * ARM (ARMv6K, as found in the 3DS) code. The translation is built to give
 * exactly the same results as the interpreter:
 *
 *  - Guest registers stay in cpu->gprs. Guest NZCV flags are kept in host r8
 *    and moved in and out of the host flags only around instructions that use
 *    them.
 *  - Simple data-processing instructions are translated natively. ARMv4 and
 *    ARMv6 compute the same results and flags for them.
 *  - Every other instruction calls the interpreter's own handler for that
 *    opcode, with cycles, flags, PC and the prefetch words synced first, so
 *    memory timing, open-bus reads, the GamePak prefetch buffer and idle loop
 *    detection all behave exactly as in the interpreter.
 *  - The cycle counter is checked against nextEvent after every guest
 *    instruction, so IRQs, timers and DMA happen at the same instruction
 *    boundaries as with the interpreter.
 *  - Writes to EWRAM/IWRAM pages that hold translated code invalidate the
 *    affected blocks; if the running block is invalidated it exits right
 *    after the store.
 *
 * Host register usage inside translated code:
 *   r4  = struct ARMCore*
 *   r5  = cpu->cycles
 *   r6  = cycles per sequential opcode fetch (1 + activeSeqCycles)
 *   r7  = cpu->nextEvent
 *   r8  = guest flags in bits 31-28
 *   r9  = host pointer to the block's guest code
 *   r10 = guest address of the block's first instruction
 *   r11 = &gba->memory.lastPrefetchedPc
 */
#ifdef JITLOG
#include <stdio.h>
#endif
#ifdef JITDBG
#include <stdio.h>
#include <stdlib.h>
static unsigned long long dbgThumbWords[256], dbgThumbCount[256], dbgArmWords[256], dbgArmCount[256], dbgTailWords;
static void _dbgDump(void) {
	unsigned i; unsigned long long tw = 0, aw = 0;
	for (i = 0; i < 256; ++i) { tw += dbgThumbWords[i]; aw += dbgArmWords[i]; }
	fprintf(stderr, "words: thumb=%llu arm=%llu tail=%llu\n", tw, aw, dbgTailWords);
	for (i = 0; i < 256; ++i) if (dbgThumbWords[i] * 100 > tw) fprintf(stderr, "  T %02X: %llu words, %llu insns, %.1f avg\n", i, dbgThumbWords[i], dbgThumbCount[i], (double) dbgThumbWords[i] / dbgThumbCount[i]);
	for (i = 0; i < 256; ++i) if (dbgArmWords[i] * 50 > aw) fprintf(stderr, "  A %02X: %llu words, %llu insns, %.1f avg\n", i, dbgArmWords[i], dbgArmCount[i], (double) dbgArmWords[i] / dbgArmCount[i]);
}
#endif
#include <mgba/internal/arm/jit.h>

#include <mgba/internal/arm/arm.h>
#include <mgba/internal/arm/isa-arm.h>
#include <mgba/internal/arm/isa-inlines.h>
#include <mgba/internal/arm/isa-thumb.h>
#include <mgba/internal/gba/gba.h>
#include <mgba-util/math.h>

#if ARM_JIT_SUPPORTED

#include <stddef.h>


#define JIT_MAX_ARM_INSNS 120
#define JIT_MAX_THUMB_INSNS 120
#define JIT_HASH_BITS 14
#define JIT_HASH_SIZE (1 << JIT_HASH_BITS)
#define JIT_MAX_BLOCKS 0x4000
#define JIT_MAX_LINKS 0x20000
#define JIT_MAX_CONSTS 256
#define JIT_MAX_BLOCK_WORDS 0x2000
#define JIT_TRAMPOLINE_WORDS 4096
#define JIT_SCRATCH_WORDS 1024
#define JIT_STEP_LIMIT 64
#define JIT_MAX_CHAIN_LINKS 0x8000
#define JIT_MAX_SLOTS 64
#define JIT_FAST_BITS 10
#define JIT_FAST_SIZE (1 << JIT_FAST_BITS)
#define JIT_HOT_PERCENT 50

enum {
	R0 = 0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, RSP, RLR, RPC
};

enum {
	COND_EQ = 0x0,
	COND_NE = 0x1,
	COND_GE = 0xA,
	COND_AL = 0xE,
	COND_NV = 0xF
};

enum {
	ALU_AND = 0x0,
	ALU_EOR = 0x1,
	ALU_SUB = 0x2,
	ALU_RSB = 0x3,
	ALU_ADD = 0x4,
	ALU_ADC = 0x5,
	ALU_SBC = 0x6,
	ALU_RSC = 0x7,
	ALU_TST = 0x8,
	ALU_TEQ = 0x9,
	ALU_CMP = 0xA,
	ALU_CMN = 0xB,
	ALU_ORR = 0xC,
	ALU_MOV = 0xD,
	ALU_BIC = 0xE,
	ALU_MVN = 0xF
};

enum {
	SHIFT_LSL = 0,
	SHIFT_LSR = 1,
	SHIFT_ASR = 2,
	SHIFT_ROR = 3
};

struct ARMJitChainLink;

struct ARMJitBlock {
	uint32_t guestAddress;
	uint8_t thumb;
	uint8_t dead;
	uint8_t ram;
	uint16_t length;
	uint32_t ramStart; // byte range in the combined EWRAM+IWRAM buffer (RAM blocks)
	uint32_t ramEnd;
	const uint8_t* hostCode;
	const uint32_t* entry; // sets up r6, r9 and r10
	const uint32_t* fastEntry; // for jumps that leave those right for this block
	struct ARMJitBlock* nextInHash;
	struct ARMJitChainLink* incoming;
};

/* A patched branch from one block's exit straight into another block */
struct ARMJitChainLink {
	uint32_t* slot;
	uint32_t original;
	struct ARMJitChainLink* next;
};

struct ARMJitPageLink {
	struct ARMJitBlock* block;
	struct ARMJitPageLink* next;
};

typedef void (*ARMJitEnter)(struct ARMCore* cpu, const uint32_t* code);

struct ARMJit {
	bool enabled;
	uint32_t* mem;
	size_t memWords;
	/* Block bodies fill [codeStart, coldStart); their rarely run parts (slow
	 * paths, exits, constants) fill [coldStart, codeEnd). Keeping the two
	 * apart packs the code that actually runs into fewer cache lines. */
	uint32_t* codeStart;
	uint32_t* codePtr;
	uint32_t* coldStart;
	uint32_t* coldPtr;
	uint32_t* codeEnd;
	ARMJitCacheFlush flushCache;

	ARMJitEnter enter;
	uint32_t* exitPlain;
	uint32_t* exitSyncArm;
	uint32_t* exitSyncThumb;
	unsigned stepping; // interpret this many steps (at least one) after a bail
	/* Shared out-of-line code, so blocks stay small (see _buildStubs) */
	uint32_t* helperStub[2]; // [thumb]
	uint32_t* memStub[2][2][8]; // [thumb][running from ROM][kind]
	uint32_t* multiPrepStub[2]; // [store]
	uint32_t* multiDoneStub[2][2][2]; // [thumb][running from ROM][load]
	uint32_t* romLiteralStub;
	uint32_t* indirectExitStub;

	struct ARMJitBlock* hash[JIT_HASH_SIZE];
	struct ARMJitBlock* blocks;
	size_t nBlocks;
	struct ARMJitPageLink* links;
	size_t nLinks;
	struct ARMJitPageLink* pageHead[ARM_JIT_RAM_PAGES];
	uint8_t pages[ARM_JIT_RAM_PAGES];
	struct ARMJitChainLink* chainLinks;
	size_t nChainLinks;

	struct ARMJitStats stats;

	/* Blocks depend on these GBA settings; a change flushes the cache */
	int idleOptimization;
	uint32_t idleLoop;

	/* Reused for every block, so compiling does not allocate */
	struct ARMJitEmitter* emitter;
	struct ARMJitEmitter* scratch;
	uint32_t* scratchBuffer;

	/* Indirect branch targets: key = guest address | thumb, value = entry */
	struct {
		uint32_t key;
		const uint32_t* entry;
	} fast[JIT_FAST_SIZE];

	int32_t offCycles;
	int32_t offNextEvent;
	int32_t offCpsr;
	int32_t offPrefetch0;
	int32_t offPrefetch1;
	int32_t offSeq32;
	int32_t offSeq16;
	int32_t offExitRequest;
	int32_t offCurrentBlock;
	int32_t offLastSlot;
	int32_t offLastSlotTarget;
	int32_t offShifterCarry;
};

struct ARMJitEmitter {
	uint32_t* start;
	uint32_t* ptr;
	uint32_t* end;
	/* The other of the two output areas (see _setCold) */
	uint32_t* otherPtr;
	uint32_t* otherEnd;
	bool cold;
	bool overflow;
	/* Where the guest NZCV flags are right now (see _flagsNeedHost etc.) */
	int flags;
	/* Constants loaded PC-relative from a pool right after the block body */
	uint32_t consts[JIT_MAX_CONSTS];
	unsigned nConsts;
	struct {
		uint32_t* site;
		unsigned index;
	} constUses[JIT_MAX_CONSTS];
	unsigned nConstUses;
	struct {
		uint32_t* site;
		unsigned insn;
	} stubFixups[JIT_MAX_ARM_INSNS + JIT_MAX_THUMB_INSNS];
	unsigned nStubFixups;
	struct {
		uint32_t* site;
		uint32_t target;
		bool synced; // PC and prefetch are already in the CPU struct
		int32_t pcAfter; // else: guest PC value to sync, relative to r10
		bool fast; // r9 and r10 were moved to the target: jump to its fastEntry
	} slots[JIT_MAX_SLOTS];
	unsigned nSlots;
	/* Event exits that sync an arbitrary guest PC value (native branches) */
	struct {
		uint32_t* site;
		int32_t pcAfter;
	} exits[JIT_MAX_ARM_INSNS + JIT_MAX_THUMB_INSNS];
	unsigned nExits;
	/* Event-check batching: the next segSkip native instructions skip their
	 * own event check, because a check before them covered them all */
	unsigned segSkip;
	struct {
		uint32_t* site;
		int32_t pcBefore;
	} bails[JIT_MAX_ARM_INSNS + JIT_MAX_THUMB_INSNS];
	unsigned nBails;
	/* Out-of-line pieces emitted after the block body */
	struct {
		uint32_t* site; // branch to patch to the out-of-line code
		uint32_t* back; // where to continue in the block
		unsigned index;
		int kind; // 0 = taken Thumb conditional branch, 1 = memory slow path/event
		uintptr_t handler;
		uint32_t target;
		bool store;
		unsigned guestReg;
		int writebackReg;
		int32_t writebackDelta;
		unsigned rewind; // r6 charges to take back: instructions of the run not executed
		bool native; // kind 0: branch without the interpreter
	} deferred[JIT_MAX_ARM_INSNS + JIT_MAX_THUMB_INSNS];
	unsigned nDeferred;
	/* Set while classifying instructions on a scratch emitter */
	bool sawNative;
	bool sawOther;
};

/* ---------------------------------------------------------------------------
 * Host instruction encoding
 * ------------------------------------------------------------------------- */

static inline void _emit(struct ARMJitEmitter* e, uint32_t insn) {
	if (e->ptr >= e->end) {
		e->overflow = true;
		return;
	}
	*e->ptr++ = insn;
}

/* Switches between emitting the block body and its out-of-line part */
static void _setCold(struct ARMJitEmitter* e, bool cold) {
	if (e->cold == cold) {
		return;
	}
	uint32_t* ptr = e->ptr;
	uint32_t* end = e->end;
	e->ptr = e->otherPtr;
	e->end = e->otherEnd;
	e->otherPtr = ptr;
	e->otherEnd = end;
	e->cold = cold;
}

static bool _encodeImm(uint32_t value, uint32_t* out) {
	int rot;
	for (rot = 0; rot < 16; ++rot) {
		uint32_t v = rot ? (value << (2 * rot)) | (value >> (32 - 2 * rot)) : value;
		if (v <= 0xFF) {
			*out = ((uint32_t) rot << 8) | v;
			return true;
		}
	}
	return false;
}

static void _dpImm(struct ARMJitEmitter* e, unsigned cond, unsigned op, bool s, unsigned rd, unsigned rn, uint32_t value) {
	uint32_t imm;
	if (!_encodeImm(value, &imm)) {
		e->overflow = true; // Programming error; refuse to produce a block
		return;
	}
	_emit(e, (cond << 28) | 0x02000000 | (op << 21) | ((uint32_t) s << 20) | (rn << 16) | (rd << 12) | imm);
}

static void _dpReg(struct ARMJitEmitter* e, unsigned cond, unsigned op, bool s, unsigned rd, unsigned rn, unsigned rm, unsigned shift, unsigned shiftImm) {
	_emit(e, (cond << 28) | (op << 21) | ((uint32_t) s << 20) | (rn << 16) | (rd << 12) | ((shiftImm & 0x1F) << 7) | (shift << 5) | rm);
}

static void _dpRegShiftReg(struct ARMJitEmitter* e, unsigned cond, unsigned op, bool s, unsigned rd, unsigned rn, unsigned rm, unsigned shift, unsigned rs) {
	_emit(e, (cond << 28) | (op << 21) | ((uint32_t) s << 20) | (rn << 16) | (rd << 12) | (rs << 8) | (shift << 5) | 0x10 | rm);
}

static void _mov(struct ARMJitEmitter* e, unsigned rd, unsigned rm) {
	_dpReg(e, COND_AL, ALU_MOV, false, rd, 0, rm, SHIFT_LSL, 0);
}

static void _memImm(struct ARMJitEmitter* e, unsigned cond, uint32_t base, unsigned rt, unsigned rn, int32_t offset) {
	uint32_t u = 0x00800000;
	if (offset < 0) {
		u = 0;
		offset = -offset;
	}
	if (offset > 0xFFF) {
		e->overflow = true;
		return;
	}
	_emit(e, (cond << 28) | base | u | (rn << 16) | (rt << 12) | (uint32_t) offset);
}

static void _ldr(struct ARMJitEmitter* e, unsigned rt, unsigned rn, int32_t offset) {
	_memImm(e, COND_AL, 0x05100000, rt, rn, offset);
}

static void _str(struct ARMJitEmitter* e, unsigned rt, unsigned rn, int32_t offset) {
	_memImm(e, COND_AL, 0x05000000, rt, rn, offset);
}

static void _strb(struct ARMJitEmitter* e, unsigned rt, unsigned rn, int32_t offset) {
	_memImm(e, COND_AL, 0x05400000, rt, rn, offset);
}

static void _ldrh(struct ARMJitEmitter* e, unsigned rt, unsigned rn, int32_t offset) {
	uint32_t u = 0x00800000;
	if (offset < 0) {
		u = 0;
		offset = -offset;
	}
	if (offset > 0xFF) {
		e->overflow = true;
		return;
	}
	_emit(e, 0xE15000B0 | u | (rn << 16) | (rt << 12) | (((uint32_t) offset & 0xF0) << 4) | ((uint32_t) offset & 0xF));
}

static void _mrsFlags(struct ARMJitEmitter* e, unsigned rd) {
	_emit(e, 0xE10F0000 | (rd << 12));
}

static void _msrFlags(struct ARMJitEmitter* e, unsigned rm) {
	_emit(e, 0xE128F000 | rm);
}

/* Guest flags live in r8 at block entry and wherever code leaves the straight
 * line (exits, calls, stubs). In between they may stay in the host flags, so
 * that runs of flag-setting instructions don't save and restore them each
 * time. FLAGS_R8: only r8 is valid. FLAGS_HOST: only the host flags are.
 * FLAGS_BOTH: both. */
enum {
	FLAGS_R8 = 0,
	FLAGS_BOTH,
	FLAGS_HOST
};

/* The next instruction reads the host flags, or keeps some of them */
static void _flagsNeedHost(struct ARMJitEmitter* e) {
	if (e->flags == FLAGS_R8) {
		_msrFlags(e, R8);
		e->flags = FLAGS_BOTH;
	}
}

/* The last instruction left the guest flags in the host flags */
static void _flagsWritten(struct ARMJitEmitter* e) {
	e->flags = FLAGS_HOST;
}

/* r8 is about to be read (or code may leave the block) */
static void _flagsToR8(struct ARMJitEmitter* e) {
	if (e->flags == FLAGS_HOST) {
		_mrsFlags(e, R8);
		e->flags = FLAGS_BOTH;
	}
}

/* The host flags are about to be overwritten by something else */
static void _flagsClobber(struct ARMJitEmitter* e) {
	_flagsToR8(e);
	e->flags = FLAGS_R8;
}

static void _blx(struct ARMJitEmitter* e, unsigned rm) {
	_emit(e, 0xE12FFF30 | rm);
}

static void _bx(struct ARMJitEmitter* e, unsigned rm) {
	_emit(e, 0xE12FFF10 | rm);
}

static uint32_t _branchWord(unsigned cond, const uint32_t* from, const uint32_t* to) {
	int32_t offset = (int32_t) (to - (from + 2));
	return (cond << 28) | 0x0A000000 | ((uint32_t) offset & 0x00FFFFFF);
}

static void _b(struct ARMJitEmitter* e, unsigned cond, const uint32_t* to) {
	if (e->ptr >= e->end) {
		e->overflow = true;
		return;
	}
	*e->ptr = _branchWord(cond, e->ptr, to);
	++e->ptr;
}

static void _bl(struct ARMJitEmitter* e, const uint32_t* to) {
	if (e->ptr >= e->end) {
		e->overflow = true;
		return;
	}
	*e->ptr = _branchWord(COND_AL, e->ptr, to) | 0x01000000;
	++e->ptr;
}

/* rd = value, for code outside blocks (no constant pool): ldr rd, [pc]; b over; .word value */
static void _ldrLiteral(struct ARMJitEmitter* e, unsigned rd, uint32_t value) {
	_emit(e, 0xE59F0000 | (rd << 12));
	_emit(e, 0xEA000000);
	_emit(e, value);
}

/* Emits a placeholder branch and returns its address for later patching */
static uint32_t* _bPlaceholder(struct ARMJitEmitter* e, unsigned cond) {
	uint32_t* site = e->ptr;
	_emit(e, cond << 28);
	return e->overflow ? NULL : site;
}

static void _patchBranch(uint32_t* site, const uint32_t* to) {
	if (site) {
		*site = _branchWord(*site >> 28, site, to);
	}
}

static void _push(struct ARMJitEmitter* e, uint16_t list) {
	_emit(e, 0xE92D0000 | list);
}

static void _pop(struct ARMJitEmitter* e, uint16_t list) {
	_emit(e, 0xE8BD0000 | list);
}

/* rd = rn + value, for any value */
static void _addImm(struct ARMJitEmitter* e, unsigned rd, unsigned rn, int32_t value) {
	uint32_t imm;
	if (_encodeImm((uint32_t) value, &imm)) {
		_dpImm(e, COND_AL, ALU_ADD, false, rd, rn, (uint32_t) value);
		return;
	}
	if (_encodeImm((uint32_t) -value, &imm)) {
		_dpImm(e, COND_AL, ALU_SUB, false, rd, rn, (uint32_t) -value);
		return;
	}
	unsigned op = ALU_ADD;
	uint32_t v = (uint32_t) value;
	if (value < 0) {
		op = ALU_SUB;
		v = (uint32_t) -value;
	}
	unsigned src = rn;
	int bit = 0;
	bool first = true;
	while (v) {
		while (!(v & (3u << bit))) {
			bit += 2;
		}
		uint32_t chunk = v & (0xFFu << bit);
		v &= ~chunk;
		_dpImm(e, COND_AL, op, false, rd, src, chunk);
		src = rd;
		first = false;
	}
	if (first) {
		_mov(e, rd, rn);
	}
}

/* How many instructions _addImm needs for VALUE */
static unsigned _addImmCost(int32_t value) {
	uint32_t imm;
	if (_encodeImm((uint32_t) value, &imm) || _encodeImm((uint32_t) -value, &imm)) {
		return 1;
	}
	uint32_t v = value < 0 ? (uint32_t) -value : (uint32_t) value;
	unsigned n = 0;
	int bit = 0;
	while (v) {
		while (!(v & (3u << bit))) {
			bit += 2;
		}
		v &= ~(0xFFu << bit);
		++n;
	}
	return n;
}

static void _loadConst(struct ARMJitEmitter* e, unsigned rd, uint32_t value) {
	uint32_t imm;
	if (_encodeImm(value, &imm)) {
		_dpImm(e, COND_AL, ALU_MOV, false, rd, 0, value);
		return;
	}
	if (_encodeImm(~value, &imm)) {
		_dpImm(e, COND_AL, ALU_MVN, false, rd, 0, ~value);
		return;
	}
	if (e->cold) {
		_ldrLiteral(e, rd, value);
		return;
	}
	unsigned i;
	for (i = 0; i < e->nConsts; ++i) {
		if (e->consts[i] == value) {
			break;
		}
	}
	if (i == e->nConsts) {
		if (e->nConsts >= JIT_MAX_CONSTS) {
			e->overflow = true;
			return;
		}
		e->consts[e->nConsts++] = value;
	}
	if (e->nConstUses >= JIT_MAX_CONSTS) {
		e->overflow = true;
		return;
	}
	e->constUses[e->nConstUses].site = e->ptr;
	e->constUses[e->nConstUses].index = i;
	++e->nConstUses;
	_emit(e, 0xE59F0000 | (rd << 12)); // ldr rd, [pc, #...], offset filled in by _emitConstPool
}

/* The pool for _loadConst, placed after the block body */
static void _emitConstPool(struct ARMJitEmitter* e) {
	uint32_t* pool = e->ptr;
	unsigned i;
	for (i = 0; i < e->nConsts; ++i) {
		_emit(e, e->consts[i]);
	}
	if (e->overflow) {
		return;
	}
	for (i = 0; i < e->nConstUses; ++i) {
		uint32_t* site = e->constUses[i].site;
		int32_t offset = (int32_t) ((pool + e->constUses[i].index - (site + 2)) * 4);
		if (offset < 0 || offset > 0xFFF) {
			e->overflow = true;
			return;
		}
		*site |= (uint32_t) offset;
	}
}

/* ---------------------------------------------------------------------------
 * Trampolines
 * ------------------------------------------------------------------------- */

static void _storeFlags(struct ARMJit* jit, struct ARMJitEmitter* e) {
	_dpReg(e, COND_AL, ALU_MOV, false, R0, 0, R8, SHIFT_LSR, 28);
	_dpReg(e, COND_AL, ALU_MOV, false, R0, 0, R0, SHIFT_LSL, 4);
	_strb(e, R0, R4, jit->offCpsr + 3);
}


/* ---------------------------------------------------------------------------
 * Shared stubs
 *
 * Blocks call these instead of carrying their own copy of the code, which
 * keeps translated code small. That matters most under 3DS emulators, where
 * every byte of new host code also has to be translated again by the
 * emulator, and a cache that overflows means recompiling everything.
 * ------------------------------------------------------------------------- */

/* ldr rd, [pc, #offset] for a literal word emitted earlier, close by */
static void _ldrNear(struct ARMJitEmitter* e, unsigned rd, const uint32_t* literal) {
	int32_t offset = (int32_t) ((const uint8_t*) literal - (const uint8_t*) (e->ptr + 2));
	_memImm(e, COND_AL, 0x05100000, rd, RPC, offset);
}

/* Interpreter fallback for one instruction.
 * In: r0 = guest PC value for the instruction. Calls the interpreter's handler
 * for the opcode there with the CPU state synced, reloads the state, and
 * returns with EQ to continue the block, or NE if the handler changed the PC
 * or the block has to be left. */
static uint32_t* _buildHelperStub(struct ARMJit* jit, struct ARMJitEmitter* e, bool thumb) {
	uint32_t* litTable = e->ptr;
	_emit(e, thumb ? (uint32_t) (uintptr_t) _thumbTable : (uint32_t) (uintptr_t) _armTable);
	uint32_t* entry = e->ptr;
	_push(e, 0x4001); // r0, lr
	_str(e, R5, R4, jit->offCycles);
	_str(e, R0, R4, offsetof(struct ARMCore, gprs) + 4 * ARM_PC);
	_dpReg(e, COND_AL, ALU_SUB, false, R1, R0, R10, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_ADD, false, R1, R1, R9, SHIFT_LSL, 0);
	if (thumb) {
		_ldrh(e, R12, R1, -2);
		_str(e, R12, R4, jit->offPrefetch0);
		_ldrh(e, R12, R1, 0);
		_str(e, R12, R4, jit->offPrefetch1);
		_ldrh(e, R1, R1, -4);
	} else {
		_ldr(e, R12, R1, -4);
		_str(e, R12, R4, jit->offPrefetch0);
		_ldr(e, R12, R1, 0);
		_str(e, R12, R4, jit->offPrefetch1);
		_ldr(e, R1, R1, -8);
	}
	_storeFlags(jit, e);
	// The same table lookup the interpreter does
	if (thumb) {
		_dpReg(e, COND_AL, ALU_MOV, false, R12, 0, R1, SHIFT_LSR, 6);
	} else {
		_dpReg(e, COND_AL, ALU_MOV, false, R12, 0, R1, SHIFT_LSR, 16);
		_dpImm(e, COND_AL, ALU_AND, false, R12, R12, 0xFF0);
		_dpReg(e, COND_AL, ALU_MOV, false, R2, 0, R1, SHIFT_LSR, 4);
		_dpImm(e, COND_AL, ALU_AND, false, R2, R2, 0xF);
		_dpReg(e, COND_AL, ALU_ORR, false, R12, R12, R2, SHIFT_LSL, 0);
	}
	_ldrNear(e, R2, litTable);
	_emit(e, 0xE7900100 | (R2 << 16) | (R2 << 12) | R12); // ldr r2, [r2, r12, lsl #2]
	_mov(e, R0, R4);
	_blx(e, R2);
	_ldr(e, R5, R4, jit->offCycles);
	_ldr(e, R7, R4, jit->offNextEvent);
	_ldr(e, R8, R4, jit->offCpsr);
	_ldr(e, R6, R4, thumb ? jit->offSeq16 : jit->offSeq32);
	_dpImm(e, COND_AL, ALU_ADD, false, R6, R6, 1);
	_pop(e, 0x4001); // r0, lr
	// Continue only if the handler left the PC alone and nothing was invalidated
	_ldr(e, R1, R4, offsetof(struct ARMCore, gprs) + 4 * ARM_PC);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R1, R0, SHIFT_LSL, 0);
	_memImm(e, COND_EQ, 0x05100000, R1, R4, jit->offExitRequest); // ldreq r1, [r4, #exitRequest]
	_dpImm(e, COND_EQ, ALU_CMP, true, 0, R1, 0);
	_bx(e, RLR);
	return entry;
}

/* kind: 0-2 = store 8/16/32, 3 = load 32, 4 = load u16, 5 = load s16,
 * 6 = load u8, 7 = load s8 */
static void _memKind(unsigned kind, bool* store, unsigned* size, bool* sign) {
	static const uint8_t sizes[8] = { 1, 2, 4, 4, 2, 2, 1, 1 };
	*store = kind < 3;
	*size = sizes[kind];
	*sign = kind == 5 || kind == 7;
}

static unsigned _memKindIndex(bool store, unsigned size, bool sign) {
	if (store) {
		return size == 1 ? 0 : size == 2 ? 1 : 2;
	}
	if (size == 4) {
		return 3;
	}
	if (size == 2) {
		return sign ? 5 : 4;
	}
	return sign ? 7 : 6;
}

static void _emitRawAccess(struct ARMJitEmitter* e, bool store, unsigned size, bool sign, unsigned rt, unsigned rn, unsigned rm) {
	if (store) {
		switch (size) {
		case 4:
			_emit(e, 0xE7800000 | (rn << 16) | (rt << 12) | rm); // str
			break;
		case 2:
			_emit(e, 0xE18000B0 | (rn << 16) | (rt << 12) | rm); // strh
			break;
		default:
			_emit(e, 0xE7C00000 | (rn << 16) | (rt << 12) | rm); // strb
			break;
		}
		return;
	}
	switch (size) {
	case 4:
		_emit(e, 0xE7900000 | (rn << 16) | (rt << 12) | rm); // ldr
		break;
	case 2:
		_emit(e, (sign ? 0xE19000F0 : 0xE19000B0) | (rn << 16) | (rt << 12) | rm); // ldrsh / ldrh
		break;
	default:
		_emit(e, (sign ? 0xE19000D0 : 0xE7D00000) | (rn << 16) | (rt << 12) | rm); // ldrsb / ldrb
		break;
	}
}

/* Access done: charge the prefetch (r6) like the block would, then return LT
 * if no event is due, or GE with r12 = 0 if one is. */
static void _emitMemStubDone(struct ARMJitEmitter* e) {
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R6, SHIFT_LSL, 0);
	_dpImm(e, COND_AL, ALU_MOV, false, R12, 0, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	_bx(e, RLR);
}

/* Charges an access that waited r3 cycles and ends the stub. The interpreter
 * adds the wait, then turns the instruction's sequential fetch into a
 * non-sequential one (N - S); with r6 = S + 1 that is wait + N + 1 in all. */
static void _emitMemStubCharge(struct ARMJitEmitter* e, bool thumb) {
	_ldr(e, R12, R4, thumb ? offsetof(struct ARMCore, memory.activeNonseqCycles16) : offsetof(struct ARMCore, memory.activeNonseqCycles32));
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R3, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R12, SHIFT_LSL, 0);
	_dpImm(e, COND_AL, ALU_ADD, false, R5, R5, 1);
	_dpImm(e, COND_AL, ALU_MOV, false, R12, 0, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	_bx(e, RLR);
}

/* GBAMemoryStall, for code running from ROM with the GamePak prefetch on.
 * In: r3 = wait, r2 = guest PC value. Charges the access and ends the stub.
 * Uses r0 and r1 as scratch around a push/pop.
 *
 * The interpreter's result is max(wait, stall) - stall - (N16 - S16), on top
 * of which the access costs N - S and the block's r6. */
static void _emitMemStubStall(struct ARMJitEmitter* e, bool thumb, const uint32_t* litLastPrefetchedPc) {
	_push(e, 0x0003); // r0, r1
	_ldrNear(e, R12, litLastPrefetchedPc);
	_ldr(e, R12, R12, 0);
	_dpReg(e, COND_AL, ALU_SUB, false, R12, R12, R2, SHIFT_LSL, 0); // distance to the last prefetch
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 16);
	_dpImm(e, 0x2, ALU_MOV, false, R12, 0, 0); // movhs
	_dpReg(e, COND_AL, ALU_MOV, false, R12, 0, R12, SHIFT_LSR, 1); // loads already made; counts up to 7
	_ldr(e, R0, R4, offsetof(struct ARMCore, memory.activeSeqCycles16)); // s
	_dpImm(e, COND_AL, ALU_ADD, false, R1, R0, 1); // stall = s + 1
	uint32_t* loop = e->ptr;
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R1, R3, SHIFT_LSL, 0);
	uint32_t* done = _bPlaceholder(e, COND_GE);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 7);
	_dpReg(e, 0xB, ALU_ADD, false, R1, R1, R0, SHIFT_LSL, 0); // addlt stall, s
	_dpImm(e, 0xB, ALU_ADD, false, R12, R12, 1); // addlt
	_b(e, 0xB, loop); // blt
	_patchBranch(done, e->ptr);
	_dpReg(e, COND_AL, ALU_ADD, false, R2, R2, R12, SHIFT_LSL, 1);
	_ldrNear(e, R12, litLastPrefetchedPc);
	_str(e, R2, R12, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R3, R1, SHIFT_LSL, 0);
	_dpReg(e, 0xC, ALU_SUB, false, R3, R3, R1, SHIFT_LSL, 0); // subgt: wait beyond the stall
	_dpReg(e, 0xC, ALU_ADD, false, R5, R5, R3, SHIFT_LSL, 0); // addgt
	if (!thumb) {
		// (N32 - S32) - (N16 - S16); r0 still holds S16
		_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R0, SHIFT_LSL, 0);
		_ldr(e, R0, R4, offsetof(struct ARMCore, memory.activeNonseqCycles16));
		_dpReg(e, COND_AL, ALU_SUB, false, R5, R5, R0, SHIFT_LSL, 0);
		_ldr(e, R0, R4, offsetof(struct ARMCore, memory.activeNonseqCycles32));
		_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R0, SHIFT_LSL, 0);
		_ldr(e, R0, R4, offsetof(struct ARMCore, memory.activeSeqCycles32));
		_dpReg(e, COND_AL, ALU_SUB, false, R5, R5, R0, SHIFT_LSL, 0);
	}
	_pop(e, 0x0003);
	_emitMemStubDone(e);
}

/* Fast EWRAM/IWRAM (and, for loads, ROM) access following GBALoad32/16/8 and
 * GBAStore32/16/8 exactly.
 * In: r0 = address, r1 = value (stores), r2 = guest PC value (ROM variant).
 * Out: LT with the access done, r5 charged (including r6) and r1 = value
 * (loads); GE with r12 = 0: the same, but an event is due; GE with r12 = 1:
 * nothing changed, the interpreter must handle the instruction.
 * Keeps r0 and r4-r11. */
static uint32_t* _buildMemStub(struct ARMJit* jit, struct ARMJitEmitter* e, struct GBA* gba, bool thumb, bool rom, unsigned kind) {
	bool store, sign;
	unsigned size;
	_memKind(kind, &store, &size, &sign);
	uint32_t* slow[8] = {0};
	unsigned nSlow = 0;
	const char* table = (size == 4) ? gba->memory.waitstatesNonseq32 : gba->memory.waitstatesNonseq16;

	// Addresses the stub needs, right before its code
	uint32_t* litPages = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) jit->pages);
	uint32_t* litWram = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) gba->memory.wram);
	uint32_t* litWait = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) table);
	uint32_t* litPrefetch = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.prefetch);
	uint32_t* litLastPrefetchedPc = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.lastPrefetchedPc);
	uint32_t* litRomSize = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.romSize);
	uint32_t* litRom = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.rom);

	uint32_t* entry = e->ptr;
	if (!store && size > 1) {
		_dpImm(e, COND_AL, ALU_TST, true, 0, R0, size - 1);
		slow[nSlow++] = _bPlaceholder(e, COND_NE);
	}
	_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R0, SHIFT_LSR, 24);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R3, GBA_REGION_IWRAM);
	uint32_t* iwram = _bPlaceholder(e, COND_EQ);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R3, GBA_REGION_EWRAM);
	uint32_t* notRam = _bPlaceholder(e, COND_NE);

	unsigned region;
	uint32_t* charge[2] = {0};
	for (region = 0; region < 2; ++region) {
		bool isIwram = region == 1;
		if (isIwram) {
			_patchBranch(iwram, e->ptr);
		}
		// r3 = offset into the combined EWRAM+IWRAM buffer
		unsigned bits = isIwram ? 17 : 14;
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R0, SHIFT_LSL, bits);
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R3, SHIFT_LSR, bits);
		if (isIwram) {
			_dpImm(e, COND_AL, ALU_ADD, false, R3, R3, ARM_JIT_EWRAM_SIZE);
		}
		if (store) {
			if (size > 1) {
				_dpImm(e, COND_AL, ALU_BIC, false, R3, R3, size - 1);
			}
			// Stores near translated code go through the interpreter
			_ldrNear(e, R12, litPages);
			_emit(e, 0xE7D00000 | (R12 << 16) | (R12 << 12) | (ARM_JIT_PAGE_SHIFT << 7) | (SHIFT_LSR << 5) | R3); // ldrb r12, [r12, r3, lsr #PAGE_SHIFT]
			_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
			slow[nSlow++] = _bPlaceholder(e, COND_NE);
		}
		_ldrNear(e, R12, litWram);
		_emitRawAccess(e, store, size, sign, R1, R12, R3);
		// Wait states: EWRAM uses the non-sequential table entry, IWRAM has none
		if (isIwram) {
			_dpImm(e, COND_AL, ALU_MOV, false, R3, 0, store ? 1 : 2);
		} else {
			_ldrNear(e, R12, litWait);
			_memImm(e, COND_AL, 0x05500000, R3, R12, GBA_REGION_EWRAM); // ldrb r3, [r12, #EWRAM]
			_dpImm(e, COND_AL, ALU_ADD, false, R3, R3, store ? 1 : 2);
			charge[0] = _bPlaceholder(e, COND_AL);
		}
	}
	_patchBranch(charge[0], e->ptr);
	if (rom) {
		// Running from ROM: the GamePak prefetch buffer may change the stall
		_ldrNear(e, R12, litPrefetch);
		_memImm(e, COND_AL, 0x05500000, R12, R12, 0); // ldrb r12, [r12]
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
		uint32_t* noStall = _bPlaceholder(e, COND_EQ);
		_emitMemStubStall(e, thumb, litLastPrefetchedPc);
		_patchBranch(noStall, e->ptr);
	}
	_emitMemStubCharge(e, thumb);

	_patchBranch(notRam, e->ptr);
	if (!store) {
		/* Cartridge ROM (0x08000000-0x0CFFFFFF), inside the ROM image: same
		 * value and wait states as GBALoad32/16/8, and no prefetch stall since
		 * the address is past the ROM base. */
		_dpImm(e, COND_AL, ALU_SUB, false, R12, R3, GBA_REGION_ROM0);
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, GBA_REGION_ROM2 - GBA_REGION_ROM0);
		slow[nSlow++] = _bPlaceholder(e, 0x8); // HI
		_ldrNear(e, R12, litWait);
		_emit(e, 0xE7D00000 | (R12 << 16) | (R12 << 12) | R3); // ldrb r12, [r12, r3]: the region's wait
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R0, SHIFT_LSL, 7);
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R3, SHIFT_LSR, 7); // offset in the 32 MiB window
		_ldrNear(e, R1, litRomSize);
		_ldr(e, R1, R1, 0);
		_dpReg(e, COND_AL, ALU_CMP, true, 0, R3, R1, SHIFT_LSL, 0);
		slow[nSlow++] = _bPlaceholder(e, 0x2); // HS
		_ldrNear(e, R1, litRom);
		_ldr(e, R1, R1, 0);
		_emitRawAccess(e, false, size, sign, R1, R1, R3);
		_dpImm(e, COND_AL, ALU_ADD, false, R3, R12, 2);
		_emitMemStubCharge(e, thumb);
	}
	// Slow: GE with r12 = 1, nothing changed
	unsigned i;
	for (i = 0; i < nSlow; ++i) {
		_patchBranch(slow[i], e->ptr);
	}
	_dpImm(e, COND_AL, ALU_MOV, false, R12, 0, 1);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
	_bx(e, RLR);
	return entry;
}

/* First half of an inline LDM/STM on EWRAM/IWRAM (see _emitFastMultiple).
 * In: r0 = lowest address, r3 = word count. Out: EQ with r1 = host pointer and
 * r12 = 1 for IWRAM, 0 for EWRAM; NE if the interpreter must do the transfer
 * (unaligned, not RAM, wrapping around the region, or a store next to
 * translated code). Keeps r2 and r3. */
static uint32_t* _buildMultiPrepStub(struct ARMJit* jit, struct ARMJitEmitter* e, struct GBA* gba, bool store) {
	uint32_t* slow[6] = {0};
	unsigned nSlow = 0;
	uint32_t* litPages = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) jit->pages);
	uint32_t* litWram = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) gba->memory.wram);

	uint32_t* entry = e->ptr;
	_dpImm(e, COND_AL, ALU_TST, true, 0, R0, 3);
	slow[nSlow++] = _bPlaceholder(e, COND_NE);
	_dpReg(e, COND_AL, ALU_MOV, false, R12, 0, R0, SHIFT_LSR, 24);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, GBA_REGION_IWRAM);
	uint32_t* iwram = _bPlaceholder(e, COND_EQ);
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, GBA_REGION_EWRAM);
	slow[nSlow++] = _bPlaceholder(e, COND_NE);
	// r1 = offset into the combined buffer, r12 = end of the region there
	_dpReg(e, COND_AL, ALU_MOV, false, R1, 0, R0, SHIFT_LSL, 14);
	_dpReg(e, COND_AL, ALU_MOV, false, R1, 0, R1, SHIFT_LSR, 14);
	_dpImm(e, COND_AL, ALU_MOV, false, R12, 0, ARM_JIT_EWRAM_SIZE);
	uint32_t* common = _bPlaceholder(e, COND_AL);
	_patchBranch(iwram, e->ptr);
	_dpReg(e, COND_AL, ALU_MOV, false, R1, 0, R0, SHIFT_LSL, 17);
	_dpReg(e, COND_AL, ALU_MOV, false, R1, 0, R1, SHIFT_LSR, 17);
	_dpImm(e, COND_AL, ALU_ADD, false, R1, R1, ARM_JIT_EWRAM_SIZE);
	_dpImm(e, COND_AL, ALU_MOV, false, R12, 0, ARM_JIT_EWRAM_SIZE + ARM_JIT_IWRAM_SIZE);
	_patchBranch(common, e->ptr);
	// The whole transfer must stay inside the region (no mirroring wraparound)
	_dpReg(e, COND_AL, ALU_ADD, false, R0, R1, R3, SHIFT_LSL, 2);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R0, R12, SHIFT_LSL, 0);
	slow[nSlow++] = _bPlaceholder(e, 0x8); // HI
	if (store) {
		// At most 16 words: the first and last word's pages cover them all
		_ldrNear(e, R12, litPages);
		_dpImm(e, COND_AL, ALU_SUB, false, R0, R0, 4);
		_emit(e, 0xE7D00000 | (R12 << 16) | (R0 << 12) | (ARM_JIT_PAGE_SHIFT << 7) | (SHIFT_LSR << 5) | R0); // ldrb r0, [r12, r0, lsr #PAGE_SHIFT]
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R0, 0);
		slow[nSlow++] = _bPlaceholder(e, COND_NE);
		_emit(e, 0xE7D00000 | (R12 << 16) | (R0 << 12) | (ARM_JIT_PAGE_SHIFT << 7) | (SHIFT_LSR << 5) | R1); // ldrb r0, [r12, r1, lsr #PAGE_SHIFT]
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R0, 0);
		slow[nSlow++] = _bPlaceholder(e, COND_NE);
	}
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R1, ARM_JIT_EWRAM_SIZE);
	_dpImm(e, 0x3, ALU_MOV, false, R12, 0, 0); // movlo
	_dpImm(e, 0x2, ALU_MOV, false, R12, 0, 1); // movhs
	_ldrNear(e, R0, litWram);
	_dpReg(e, COND_AL, ALU_ADD, false, R1, R0, R1, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R0, R0, SHIFT_LSL, 0); // EQ
	_bx(e, RLR);
	unsigned i;
	for (i = 0; i < nSlow; ++i) {
		_patchBranch(slow[i], e->ptr);
	}
	_dpImm(e, COND_AL, ALU_MOV, true, R0, 0, 1); // NE
	_bx(e, RLR);
	return entry;
}

/* Second half: charges the transfer like GBALoadMultiple/GBAStoreMultiple.
 * In: r3 = word count, r12 = 1 for IWRAM, r2 = guest PC value (ROM variant).
 * Wait: IWRAM = count; EWRAM = (S - N) + count * (S + 1); loads add one more.
 * Out: LT, or GE if an event is due. */
static uint32_t* _buildMultiDoneStub(struct ARMJitEmitter* e, struct GBA* gba, bool thumb, bool rom, bool load) {
	uint32_t* litSeq = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.waitstatesSeq32[GBA_REGION_EWRAM]);
	uint32_t* litNonseq = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.waitstatesNonseq32[GBA_REGION_EWRAM]);
	uint32_t* litPrefetch = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.prefetch);
	uint32_t* litLastPrefetchedPc = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) &gba->memory.lastPrefetchedPc);

	uint32_t* entry = e->ptr;
	_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
	uint32_t* iwram = _bPlaceholder(e, COND_NE);
	_ldrNear(e, R12, litSeq);
	_memImm(e, COND_AL, 0x05500000, R0, R12, 0); // ldrb r0, [r12] (S)
	_ldrNear(e, R12, litNonseq);
	_memImm(e, COND_AL, 0x05500000, R1, R12, 0); // ldrb r1, [r12] (N)
	_dpImm(e, COND_AL, ALU_ADD, false, R12, R3, 1);
	_emit(e, 0xE0000090 | (R0 << 16) | (R12 << 8) | R0); // mul r0, r0, r12
	_dpReg(e, COND_AL, ALU_SUB, false, R0, R0, R1, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_ADD, false, R3, R3, R0, SHIFT_LSL, 0);
	_patchBranch(iwram, e->ptr);
	if (load) {
		_dpImm(e, COND_AL, ALU_ADD, false, R3, R3, 1);
	}
	if (rom) {
		_ldrNear(e, R12, litPrefetch);
		_memImm(e, COND_AL, 0x05500000, R12, R12, 0); // ldrb r12, [r12]
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
		uint32_t* noStall = _bPlaceholder(e, COND_EQ);
		_emitMemStubStall(e, thumb, litLastPrefetchedPc);
		_patchBranch(noStall, e->ptr);
	}
	_emitMemStubCharge(e, thumb);
	return entry;
}

/* Charges a Thumb PC-relative load from ROM whose value is a constant.
 * In: r0 = region of the literal. The interpreter adds that region's 32-bit
 * wait + 2 and turns the fetch non-sequential; with r6 that is wait + N + 3.
 * Out: LT, or GE if an event is due. */
static uint32_t* _buildRomLiteralStub(struct ARMJitEmitter* e, struct GBA* gba) {
	uint32_t* litWait = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) gba->memory.waitstatesNonseq32);
	uint32_t* entry = e->ptr;
	_ldrNear(e, R12, litWait);
	_emit(e, 0xE7D00000 | (R12 << 16) | (R0 << 12) | R0); // ldrb r0, [r12, r0]
	_ldr(e, R12, R4, offsetof(struct ARMCore, memory.activeNonseqCycles16));
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R0, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R12, SHIFT_LSL, 0);
	_dpImm(e, COND_AL, ALU_ADD, false, R5, R5, 3);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	_bx(e, RLR);
	return entry;
}

/* After an instruction that changed the PC to a value only known at run time:
 * look the new PC up in the fast table and jump to its block, or leave. */
static uint32_t* _buildIndirectExitStub(struct ARMJit* jit, struct ARMJitEmitter* e) {
	uint32_t* litFast = e->ptr;
	_emit(e, (uint32_t) (uintptr_t) jit->fast);
	uint32_t* entry = e->ptr;
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	_b(e, COND_GE, jit->exitPlain);
	_ldr(e, R0, R4, offsetof(struct ARMCore, gprs) + 4 * ARM_PC);
	_ldr(e, R1, R4, offsetof(struct ARMCore, executionMode));
	_dpImm(e, COND_AL, ALU_SUB, false, R0, R0, 4);
	_dpReg(e, COND_AL, ALU_ADD, false, R0, R0, R1, SHIFT_LSL, 1); // Thumb: PC - 2
	_dpReg(e, COND_AL, ALU_ORR, false, R0, R0, R1, SHIFT_LSL, 0); // key
	_dpReg(e, COND_AL, ALU_MOV, false, R2, 0, R0, SHIFT_LSL, 32 - (JIT_FAST_BITS + 1));
	_dpReg(e, COND_AL, ALU_MOV, false, R2, 0, R2, SHIFT_LSR, 32 - (JIT_FAST_BITS + 1) - 2);
	_dpImm(e, COND_AL, ALU_BIC, false, R2, R2, 7); // index * 8
	_ldrNear(e, R3, litFast);
	_dpReg(e, COND_AL, ALU_ADD, false, R3, R3, R2, SHIFT_LSL, 0);
	_ldr(e, R12, R3, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R12, R0, SHIFT_LSL, 0);
	_b(e, COND_NE, jit->exitPlain);
	_ldr(e, R12, R3, 4);
	_bx(e, R12);
	return entry;
}

static bool _buildTrampolines(struct ARMJit* jit, struct GBA* gba) {
	struct ARMJitEmitter e = {0};
	e.start = jit->mem;
	e.ptr = jit->mem;
	e.end = jit->mem + JIT_TRAMPOLINE_WORDS;

	// enter(cpu, code)
	uint32_t* litLastPrefetchedPc = e.ptr;
	_emit(&e, (uint32_t) (uintptr_t) &gba->memory.lastPrefetchedPc);
	jit->enter = (ARMJitEnter) (uintptr_t) e.ptr;
	_push(&e, 0x5FF0); // r4-r12, lr
	_mov(&e, R4, R0);
	_ldrNear(&e, R11, litLastPrefetchedPc);
	// Only blocks in RAM name themselves; anything non-NULL means "in a block"
	_str(&e, R4, R4, jit->offCurrentBlock);
	_ldr(&e, R5, R4, jit->offCycles);
	_ldr(&e, R7, R4, jit->offNextEvent);
	_ldr(&e, R8, R4, jit->offCpsr);
	_bx(&e, R1);

	// exitPlain: cycles and flags back to the CPU struct, return
	jit->exitPlain = e.ptr;
	_str(&e, R5, R4, jit->offCycles);
	_storeFlags(jit, &e);
	_pop(&e, 0x9FF0); // r4-r12, pc

	// exitSyncArm: r0 = PC value after the last executed instruction
	jit->exitSyncArm = e.ptr;
	_str(&e, R0, R4, offsetof(struct ARMCore, gprs) + 4 * ARM_PC);
	_dpReg(&e, COND_AL, ALU_SUB, false, R1, R0, R10, SHIFT_LSL, 0);
	_dpReg(&e, COND_AL, ALU_ADD, false, R1, R1, R9, SHIFT_LSL, 0);
	_ldr(&e, R2, R1, -4);
	_str(&e, R2, R4, jit->offPrefetch0);
	_ldr(&e, R2, R1, 0);
	_str(&e, R2, R4, jit->offPrefetch1);
	_b(&e, COND_AL, jit->exitPlain);

	// exitSyncThumb
	jit->exitSyncThumb = e.ptr;
	_str(&e, R0, R4, offsetof(struct ARMCore, gprs) + 4 * ARM_PC);
	_dpReg(&e, COND_AL, ALU_SUB, false, R1, R0, R10, SHIFT_LSL, 0);
	_dpReg(&e, COND_AL, ALU_ADD, false, R1, R1, R9, SHIFT_LSL, 0);
	_ldrh(&e, R2, R1, -2);
	_str(&e, R2, R4, jit->offPrefetch0);
	_ldrh(&e, R2, R1, 0);
	_str(&e, R2, R4, jit->offPrefetch1);
	_b(&e, COND_AL, jit->exitPlain);

	unsigned thumb;
	for (thumb = 0; thumb < 2; ++thumb) {
		jit->helperStub[thumb] = _buildHelperStub(jit, &e, thumb);
		unsigned rom, kind;
		for (rom = 0; rom < 2; ++rom) {
			for (kind = 0; kind < 8; ++kind) {
				jit->memStub[thumb][rom][kind] = _buildMemStub(jit, &e, gba, thumb, rom, kind);
			}
			jit->multiDoneStub[thumb][rom][0] = _buildMultiDoneStub(&e, gba, thumb, rom, false);
			jit->multiDoneStub[thumb][rom][1] = _buildMultiDoneStub(&e, gba, thumb, rom, true);
		}
	}
	jit->multiPrepStub[0] = _buildMultiPrepStub(jit, &e, gba, false);
	jit->multiPrepStub[1] = _buildMultiPrepStub(jit, &e, gba, true);
	jit->romLiteralStub = _buildRomLiteralStub(&e, gba);
	jit->indirectExitStub = _buildIndirectExitStub(jit, &e);

	return !e.overflow;
}

/* ---------------------------------------------------------------------------
 * Block bookkeeping
 * ------------------------------------------------------------------------- */

static inline unsigned _hash(uint32_t address, bool thumb) {
	uint32_t h = (address >> 1) ^ (address >> (JIT_HASH_BITS + 1)) ^ (address >> 24) ^ (thumb ? 0x2545 : 0);
	return h & (JIT_HASH_SIZE - 1);
}

static struct ARMJitBlock* _lookup(struct ARMJit* jit, uint32_t address, bool thumb) {
	struct ARMJitBlock* block = jit->hash[_hash(address, thumb)];
	while (block) {
		if (block->guestAddress == address && block->thumb == thumb) {
			return block;
		}
		block = block->nextInHash;
	}
	return NULL;
}

static void _unlinkFromHash(struct ARMJit* jit, struct ARMJitBlock* block) {
	struct ARMJitBlock** slot = &jit->hash[_hash(block->guestAddress, block->thumb)];
	while (*slot) {
		if (*slot == block) {
			*slot = block->nextInHash;
			block->nextInHash = NULL;
			return;
		}
		slot = &(*slot)->nextInHash;
	}
}

static void _flush(struct ARMJit* jit, struct ARMCore* cpu) {
	memset(jit->hash, 0, sizeof(jit->hash));
	memset(jit->pageHead, 0, sizeof(jit->pageHead));
	memset(jit->pages, 0, sizeof(jit->pages));
	memset(jit->fast, 0, sizeof(jit->fast));
	jit->nBlocks = 0;
	jit->nLinks = 0;
	jit->nChainLinks = 0;
	jit->codePtr = jit->codeStart;
	jit->coldPtr = jit->coldStart;
	if (cpu->jitCurrentBlock) {
		cpu->jitExitRequest = 1;
	}
	cpu->jitLastSlot = NULL;
	jit->stepping = 0;
	++jit->stats.flushes;
}

/* Region in the combined EWRAM+IWRAM buffer, or -1 if the address is not RAM */
static int32_t _ramOffset(uint32_t address) {
	switch (address >> 24) {
	case GBA_REGION_EWRAM:
		return address & (ARM_JIT_EWRAM_SIZE - 1);
	case GBA_REGION_IWRAM:
		return ARM_JIT_EWRAM_SIZE + (address & (ARM_JIT_IWRAM_SIZE - 1));
	default:
		return -1;
	}
}

static bool _registerPages(struct ARMJit* jit, struct ARMJitBlock* block, uint32_t bytes) {
	int32_t start = _ramOffset(block->guestAddress);
	if (start < 0) {
		return true;
	}
	block->ramStart = (uint32_t) start;
	block->ramEnd = (uint32_t) start + bytes;
	unsigned first = (unsigned) start >> ARM_JIT_PAGE_SHIFT;
	unsigned last = (unsigned) (start + bytes - 1) >> ARM_JIT_PAGE_SHIFT;
	unsigned page;
	for (page = first; page <= last && page < ARM_JIT_RAM_PAGES; ++page) {
		if (jit->nLinks >= JIT_MAX_LINKS) {
			return false;
		}
		struct ARMJitPageLink* link = &jit->links[jit->nLinks++];
		link->block = block;
		link->next = jit->pageHead[page];
		jit->pageHead[page] = link;
		jit->pages[page] = 1;
	}
	return true;
}

/* ---------------------------------------------------------------------------
 * Translation helpers shared by ARM and Thumb
 * ------------------------------------------------------------------------- */

struct ARMJitInsnContext {
	struct ARMJit* jit;
	struct ARMJitEmitter* e;
	struct ARMJitBlock* block;
	struct GBA* gba;
	unsigned index;
	bool thumb;
	bool prevWasBL1;
	uint32_t bl1Value;
};

static bool _sameMapping(const struct ARMJitInsnContext* ctx, uint32_t target);

static int32_t _pcValue(const struct ARMJitInsnContext* ctx) {
	// Value of the guest PC register while the instruction executes
	return ctx->thumb ? (int32_t) (ctx->index * 2 + 4) : (int32_t) (ctx->index * 4 + 8);
}

static void _cyclesAndCheck(struct ARMJitInsnContext* ctx) {
	struct ARMJitEmitter* e = ctx->e;
	e->sawOther = true;
	e->segSkip = 0;
	_flagsClobber(e);
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R6, SHIFT_LSL, 0);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	if (e->nStubFixups < sizeof(e->stubFixups) / sizeof(e->stubFixups[0])) {
		e->stubFixups[e->nStubFixups].site = _bPlaceholder(e, COND_GE);
		e->stubFixups[e->nStubFixups].insn = ctx->index;
		++e->nStubFixups;
	} else {
		e->overflow = true;
	}
}

/* Leaves at the end of this instruction when the compare just made (by a
 * stub) found an event due */
static void _eventExitIfGE(struct ARMJitInsnContext* ctx) {
	struct ARMJitEmitter* e = ctx->e;
	e->sawOther = true;
	e->segSkip = 0;
	if (e->nStubFixups < sizeof(e->stubFixups) / sizeof(e->stubFixups[0])) {
		e->stubFixups[e->nStubFixups].site = _bPlaceholder(e, COND_GE);
		e->stubFixups[e->nStubFixups].insn = ctx->index;
		++e->nStubFixups;
	} else {
		e->overflow = true;
	}
}

static void _eventCheckOnly(struct ARMJitInsnContext* ctx) {
	struct ARMJitEmitter* e = ctx->e;
	e->sawOther = true;
	e->segSkip = 0;
	_flagsClobber(e);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	if (e->nStubFixups < sizeof(e->stubFixups) / sizeof(e->stubFixups[0])) {
		e->stubFixups[e->nStubFixups].site = _bPlaceholder(e, COND_GE);
		e->stubFixups[e->nStubFixups].insn = ctx->index;
		++e->nStubFixups;
	} else {
		e->overflow = true;
	}
}

/* End of a native instruction whose cost is r6 plus at most one extra cycle
 * (already added). Inside a checked segment the event check is skipped. */
static void _nativeCycles(struct ARMJitInsnContext* ctx) {
	struct ARMJitEmitter* e = ctx->e;
	if (e->segSkip) {
		// The run's check already charged this instruction's r6
		e->sawNative = true;
		return;
	}
	_cyclesAndCheck(ctx);
	e->sawOther = false;
	e->sawNative = true;
}

static void _loadGuestReg(struct ARMJitInsnContext* ctx, unsigned hostReg, unsigned guestReg, int32_t pcExtra) {
	if (guestReg == ARM_PC) {
		_addImm(ctx->e, hostReg, R10, _pcValue(ctx) + pcExtra);
	} else {
		_ldr(ctx->e, hostReg, R4, offsetof(struct ARMCore, gprs) + 4 * guestReg);
	}
}

static void _storeGuestReg(struct ARMJitInsnContext* ctx, unsigned hostReg, unsigned guestReg) {
	_str(ctx->e, hostReg, R4, offsetof(struct ARMCore, gprs) + 4 * guestReg);
}

/* Emits a branch that initially leaves the block and is later patched to jump
 * straight into the block for TARGET. */
static void _emitChainSlot(struct ARMJitInsnContext* ctx, uint32_t target, bool synced, int32_t pcAfter) {
	struct ARMJitEmitter* e = ctx->e;
	e->sawOther = true;
	e->segSkip = 0;
	_flagsClobber(e);
	if (e->nSlots >= JIT_MAX_SLOTS) {
		if (synced) {
			_b(e, COND_AL, ctx->jit->exitPlain);
		} else {
			_addImm(e, R0, R10, pcAfter);
			_b(e, COND_AL, ctx->thumb ? ctx->jit->exitSyncThumb : ctx->jit->exitSyncArm);
		}
		return;
	}
	/* A jump within the mapping can make r9 and r10 right for the target here,
	 * so that the target need not load them (see _compile) */
	bool fast = false;
	int32_t delta = (int32_t) (target - ctx->block->guestAddress);
	if (!synced && _sameMapping(ctx, target) && _addImmCost(delta) <= 2) {
		fast = true;
		_addImm(e, R10, R10, delta);
		_addImm(e, R9, R9, delta);
		pcAfter -= delta;
	}
	e->slots[e->nSlots].site = _bPlaceholder(e, COND_AL);
	e->slots[e->nSlots].target = target | (ctx->thumb ? 1 : 0);
	e->slots[e->nSlots].synced = synced;
	e->slots[e->nSlots].pcAfter = pcAfter;
	e->slots[e->nSlots].fast = fast;
	++e->nSlots;
}

/* After an instruction that changed the PC to a value only known at run time
 * (BX, POP {pc}, MOV pc, ...), look the new PC up in the fast table and jump
 * straight into its block. The handler already synced PC and prefetch. */
static void _emitIndirectExit(struct ARMJitInsnContext* ctx) {
	struct ARMJitEmitter* e = ctx->e;
	_flagsClobber(e);
	_b(e, COND_AL, ctx->jit->indirectExitStub);
}

/* A branch to a fixed TARGET can skip the interpreter when all its handler
 * would do is move the PC within the mapped region: same region and mirror,
 * inside the ROM image, and not the idle loop. While mGBA is still looking for
 * an idle loop, every jump feeds that detection, so the handler stays. */
static bool _canBranchNatively(const struct ARMJitInsnContext* ctx, uint32_t target) {
	struct GBA* gba = ctx->gba;
	uint32_t from = ctx->block->guestAddress;
	unsigned width = ctx->thumb ? 2 : 4;
	if (gba->idleOptimization == IDLE_LOOP_DETECT) {
		return false;
	}
	if (gba->idleOptimization >= IDLE_LOOP_REMOVE && target == gba->idleLoop) {
		return false;
	}
	return _sameMapping(ctx, target);
}

/* Whether TARGET is in the same region and mirror as the block and inside the
 * mapped memory, so that r9 and r10 moved by the distance are right for it */
static bool _sameMapping(const struct ARMJitInsnContext* ctx, uint32_t target) {
	struct GBA* gba = ctx->gba;
	uint32_t from = ctx->block->guestAddress;
	unsigned width = ctx->thumb ? 2 : 4;
	if (target & (width - 1)) {
		return false;
	}
	uint32_t mask;
	switch (from >> 24) {
	case GBA_REGION_EWRAM:
		mask = GBA_SIZE_EWRAM - 1;
		break;
	case GBA_REGION_IWRAM:
		mask = GBA_SIZE_IWRAM - 1;
		break;
	case GBA_REGION_ROM0:
	case GBA_REGION_ROM0_EX:
	case GBA_REGION_ROM1:
	case GBA_REGION_ROM1_EX:
	case GBA_REGION_ROM2:
	case GBA_REGION_ROM2_EX:
		if ((target >> 24) != (from >> 24)) {
			return false;
		}
		return (target & (GBA_SIZE_ROM0 - 1)) + 3 * width <= gba->memory.romSize;
	default:
		return false;
	}
	// RAM: the same mirror, and the prefetch must not wrap around its end
	if ((target & ~mask) != (from & ~mask)) {
		return false;
	}
	return (target & mask) + 3 * width <= mask + 1;
}

/* Takes back COUNT instructions' worth of r6 cycles charged up front */
static void _emitRewind(struct ARMJitEmitter* e, unsigned count) {
	unsigned bit;
	for (bit = 0; count >> bit; ++bit) {
		if (count & (1u << bit)) {
			_dpReg(e, COND_AL, ALU_SUB, false, R5, R5, R6, SHIFT_LSL, bit);
		}
	}
}

/* Branch to TARGET the way the interpreter's B/BL handlers do, minus the call:
 * reset the GamePak prefetch tracking, charge this instruction's prefetch plus
 * the refill of the pipeline (2 + N + S), then check for events and leave
 * through a chain slot. link != 0 is stored in LR first. */
static void _emitNativeBranch(struct ARMJitInsnContext* ctx, uint32_t target, uint32_t link) {
	struct ARMJitEmitter* e = ctx->e;
	struct GBA* gba = ctx->gba;
	unsigned width = ctx->thumb ? 2 : 4;
	int32_t pcAfter = (int32_t) (target + width - ctx->block->guestAddress);
	UNUSED(gba);
	if (link) {
		_addImm(e, R0, R10, (int32_t) (link - ctx->block->guestAddress));
		_storeGuestReg(ctx, R0, ARM_LR);
	}
	_dpImm(e, COND_AL, ALU_MOV, false, R0, 0, 0);
	_str(e, R0, R11, 0);
	_ldr(e, R0, R4, ctx->thumb ? offsetof(struct ARMCore, memory.activeNonseqCycles16) : offsetof(struct ARMCore, memory.activeNonseqCycles32));
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R6, SHIFT_LSL, 1);
	_dpImm(e, COND_AL, ALU_ADD, false, R0, R0, 1);
	_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R0, SHIFT_LSL, 0);
	_flagsClobber(e);
	_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
	if (e->nExits < sizeof(e->exits) / sizeof(e->exits[0])) {
		e->exits[e->nExits].site = _bPlaceholder(e, COND_GE);
		e->exits[e->nExits].pcAfter = pcAfter;
		++e->nExits;
	} else {
		e->overflow = true;
	}
	_emitChainSlot(ctx, target, false, pcAfter);
}

enum {
	CHAIN_NONE = 0,
	CHAIN_ALWAYS, // A static branch: leaves the block whenever it executes
	CHAIN_IF_TAKEN // A Thumb conditional branch
};

/* Calls the interpreter's handler for this instruction. Returns false on error.
 * cond is the guest condition (COND_AL for Thumb). branchClass instructions
 * always leave the block after executing. */
static void _emitHelperCall(struct ARMJitInsnContext* ctx, unsigned cond, uintptr_t handler, bool branchClass, int chain, uint32_t chainTarget) {
	struct ARMJit* jit = ctx->jit;
	struct ARMJitEmitter* e = ctx->e;
	e->sawOther = true;
	e->segSkip = 0;
	uint32_t* skip = NULL;
	int32_t pc = _pcValue(ctx);

	if (cond == COND_NV) {
		// Never executes on ARMv4
		_cyclesAndCheck(ctx);
		return;
	}
	_flagsToR8(e);
	if (cond != COND_AL) {
		_flagsNeedHost(e);
		skip = _bPlaceholder(e, cond ^ 1);
	}
	// The handler gets the flags from r8 and r8 is reloaded after it
	e->flags = FLAGS_R8;

	_addImm(e, R0, R10, pc);
	_bl(e, jit->helperStub[ctx->thumb]);
	if (chain == CHAIN_ALWAYS) {
		_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
		_b(e, COND_GE, jit->exitPlain);
		_emitChainSlot(ctx, chainTarget, true, 0);
	} else if (branchClass) {
		_emitIndirectExit(ctx);
	} else if (chain == CHAIN_IF_TAKEN) {
		// The stub returns NE when the handler changed the PC: branch taken
		uint32_t* notTaken = _bPlaceholder(e, COND_EQ);
		_dpReg(e, COND_AL, ALU_CMP, true, 0, R5, R7, SHIFT_LSL, 0);
		_b(e, COND_GE, jit->exitPlain);
		_emitChainSlot(ctx, chainTarget, true, 0);
		_patchBranch(notTaken, e->ptr);
	} else {
		_b(e, COND_NE, jit->exitPlain);
	}

	if (skip) {
		uint32_t* join = _bPlaceholder(e, COND_AL);
		_patchBranch(skip, e->ptr);
		_dpReg(e, COND_AL, ALU_ADD, false, R5, R5, R6, SHIFT_LSL, 0);
		_patchBranch(join, e->ptr);
	}
	_eventCheckOnly(ctx);
}


/* ---------------------------------------------------------------------------
 * Inline EWRAM/IWRAM loads and stores
 *
 * These follow GBALoad32/16/8 and GBAStore32/16/8 exactly for RAM addresses: the same value,
 * the same wait states (including the GamePak prefetch stall when running
 * from ROM) and the same post-access cycles. Anything else (other regions,
 * unaligned loads, stores to pages holding translated code) takes the
 * interpreter's handler instead.
 * ------------------------------------------------------------------------- */

struct ARMJitFastMem {
	bool store;
	bool sign;
	unsigned size;
	unsigned guestReg;
	int writebackReg; // -1 for none; new base value is in R0 + writebackDelta
	int32_t writebackDelta;
};

static void _memReg(struct ARMJitEmitter* e, uint32_t base, unsigned rt, unsigned rn, unsigned rm, unsigned shift, unsigned shiftImm) {
	_emit(e, (COND_AL << 28) | base | (rn << 16) | (rt << 12) | ((shiftImm & 0x1F) << 7) | (shift << 5) | rm);
}

/* Emits fast path + interpreter fallback for a memory instruction whose
 * address is computed by ADDRESS into R0. */
static void _emitMemoryWithFallback(struct ARMJitInsnContext* ctx, const struct ARMJitFastMem* m, uintptr_t handler) {
	struct ARMJitEmitter* e = ctx->e;
	bool rom = ctx->block->guestAddress >> 24 >= GBA_REGION_ROM0;
	e->sawOther = true;
	e->segSkip = 0;
	if (e->nDeferred >= sizeof(e->deferred) / sizeof(e->deferred[0])) {
		e->overflow = true;
		return;
	}
	_flagsClobber(e);
	if (m->store) {
		_loadGuestReg(ctx, R1, m->guestReg, 0);
	}
	if (rom) {
		_addImm(e, R2, R10, _pcValue(ctx));
	}
	_bl(e, ctx->jit->memStub[ctx->thumb][rom][_memKindIndex(m->store, m->size, m->sign)]);
	uint32_t* site = _bPlaceholder(e, COND_GE);
	if (!m->store) {
		_storeGuestReg(ctx, R1, m->guestReg);
	}
	if (m->writebackReg >= 0) {
		_addImm(e, R0, R0, m->writebackDelta);
		_storeGuestReg(ctx, R0, m->writebackReg);
	}
	e->deferred[e->nDeferred].site = site;
	e->deferred[e->nDeferred].back = e->ptr;
	e->deferred[e->nDeferred].index = ctx->index;
	e->deferred[e->nDeferred].kind = 1;
	e->deferred[e->nDeferred].handler = handler;
	e->deferred[e->nDeferred].store = m->store;
	e->deferred[e->nDeferred].guestReg = m->guestReg;
	e->deferred[e->nDeferred].writebackReg = m->writebackReg;
	e->deferred[e->nDeferred].writebackDelta = m->writebackDelta;
	++e->nDeferred;
}

/* Thumb PC-relative load from ROM: the value is a constant */
static void _emitThumbRomLiteral(struct ARMJitInsnContext* ctx, unsigned rd, uint32_t address, uint32_t value) {
	struct ARMJitEmitter* e = ctx->e;
	_flagsClobber(e);
	uint32_t offset = address - ctx->block->guestAddress;
	if (offset <= 0xFFC && _sameMapping(ctx, address)) {
		_ldr(e, R1, R9, (int32_t) offset); // straight from the ROM image
	} else {
		_loadConst(e, R1, value);
	}
	_storeGuestReg(ctx, R1, rd);
	_dpImm(e, COND_AL, ALU_MOV, false, R0, 0, address >> 24);
	_bl(e, ctx->jit->romLiteralStub);
	_eventExitIfGE(ctx);
}

static bool _thumbEmitMemory(struct ARMJitInsnContext* ctx, uint16_t opcode, uintptr_t handler) {
	struct ARMJitEmitter* e = ctx->e;
	struct ARMJitFastMem m = { .writebackReg = -1 };
	unsigned rd = opcode & 7;
	unsigned rn = (opcode >> 3) & 7;

	if ((opcode & 0xE000) == 0x6000 || (opcode & 0xF000) == 0x8000) {
		// LDR/STR/LDRB/STRB/LDRH/STRH rd, [rn, #imm5]
		unsigned imm = (opcode >> 6) & 0x1F;
		if ((opcode & 0xF000) == 0x8000) {
			m.size = 2;
		} else {
			m.size = (opcode & 0x1000) ? 1 : 4;
		}
		m.store = !(opcode & 0x0800);
		m.guestReg = rd;
		_loadGuestReg(ctx, R0, rn, 0);
		if (imm) {
			_dpImm(e, COND_AL, ALU_ADD, false, R0, R0, imm * m.size);
		}
		_emitMemoryWithFallback(ctx, &m, handler);
		return true;
	}
	if ((opcode & 0xF000) == 0x5000) {
		// Register offset forms
		unsigned rm = (opcode >> 6) & 7;
		switch ((opcode >> 9) & 7) {
		case 0: m.store = true; m.size = 4; break; // STR
		case 1: m.store = true; m.size = 2; break; // STRH
		case 2: m.store = true; m.size = 1; break; // STRB
		case 3: m.size = 1; m.sign = true; break; // LDRSB
		case 4: m.size = 4; break; // LDR
		case 5: m.size = 2; break; // LDRH
		case 6: m.size = 1; break; // LDRB
		case 7: m.size = 2; m.sign = true; break; // LDRSH
		}
		m.guestReg = rd;
		_loadGuestReg(ctx, R0, rn, 0);
		_loadGuestReg(ctx, R1, rm, 0);
		_dpReg(e, COND_AL, ALU_ADD, false, R0, R0, R1, SHIFT_LSL, 0);
		_emitMemoryWithFallback(ctx, &m, handler);
		return true;
	}
	if ((opcode & 0xF000) == 0x9000) {
		// LDR/STR rd, [sp, #imm8 * 4]
		m.store = !(opcode & 0x0800);
		m.size = 4;
		m.guestReg = (opcode >> 8) & 7;
		_loadGuestReg(ctx, R0, ARM_SP, 0);
		if (opcode & 0xFF) {
			_dpImm(e, COND_AL, ALU_ADD, false, R0, R0, (opcode & 0xFF) << 2);
		}
		_emitMemoryWithFallback(ctx, &m, handler);
		return true;
	}
	if ((opcode & 0xF800) == 0x4800) {
		// LDR rd, [pc, #imm8 * 4]
		unsigned reg = (opcode >> 8) & 7;
		uint32_t address = ((ctx->block->guestAddress + (uint32_t) _pcValue(ctx)) & ~3u) + ((opcode & 0xFF) << 2);
		struct GBA* gba = ctx->gba;
		if (address >> 24 >= GBA_REGION_ROM0 && address >> 24 <= GBA_REGION_ROM2_EX &&
		    ctx->block->guestAddress >> 24 >= GBA_REGION_ROM0 &&
		    (address & (GBA_SIZE_ROM0 - 4)) + 4 <= gba->memory.romSize) {
			uint32_t value;
			LOAD_32(value, address & (GBA_SIZE_ROM0 - 4), gba->memory.rom);
			_emitThumbRomLiteral(ctx, reg, address, value);
			return true;
		}
		m.size = 4;
		m.guestReg = reg;
		_addImm(e, R0, R10, (int32_t) (address - ctx->block->guestAddress));
		_emitMemoryWithFallback(ctx, &m, handler);
		return true;
	}
	return false;
}


/* Inline LDM/STM (and Thumb PUSH/POP/LDMIA/STMIA) on EWRAM/IWRAM, following
 * GBALoadMultiple/GBAStoreMultiple: sequential wait states per word, one extra
 * cycle for loads, the prefetch stall from ROM, and the handler's N-S cycles.
 * mode: 0 = IA, 1 = IB, 2 = DA, 3 = DB. */
static void _emitFastMultiple(struct ARMJitInsnContext* ctx, bool load, unsigned rn, uint16_t mask, unsigned mode,
                              bool writeback, uintptr_t handler) {
	struct ARMJitEmitter* e = ctx->e;
	struct ARMJit* jit = ctx->jit;
	bool rom = ctx->block->guestAddress >> 24 >= GBA_REGION_ROM0;
	unsigned count = popcount32(mask);
	int32_t start;
	int32_t delta;
	switch (mode) {
	case 0: start = 0; delta = 4 * count; break;
	case 1: start = 4; delta = 4 * count; break;
	case 2: start = 4 - 4 * (int32_t) count; delta = -4 * (int32_t) count; break;
	default: start = -4 * (int32_t) count; delta = -4 * (int32_t) count; break;
	}
	if (e->nDeferred >= sizeof(e->deferred) / sizeof(e->deferred[0])) {
		e->overflow = true;
		return;
	}

	_flagsClobber(e);
	_loadGuestReg(ctx, R0, rn, 0);
	if (start) {
		_addImm(e, R0, R0, start);
	}
	_dpImm(e, COND_AL, ALU_MOV, false, R3, 0, count);
	if (rom) {
		_addImm(e, R2, R10, _pcValue(ctx)); // for the prefetch stall
	}
	_bl(e, jit->multiPrepStub[!load]);
	uint32_t* slow = _bPlaceholder(e, COND_NE);

	unsigned reg;
	int32_t offset = 0;
	for (reg = 0; reg < 16; ++reg) {
		if (!(mask & (1 << reg))) {
			continue;
		}
		if (load) {
			_ldr(e, R0, R1, offset);
			_storeGuestReg(ctx, R0, reg);
		} else {
			_loadGuestReg(ctx, R0, reg, 0);
			_str(e, R0, R1, offset);
		}
		offset += 4;
	}
	if (writeback) {
		_loadGuestReg(ctx, R0, rn, 0);
		_addImm(e, R0, R0, delta);
		_storeGuestReg(ctx, R0, rn);
	}
	_bl(e, jit->multiDoneStub[ctx->thumb][rom][load]);
	_eventExitIfGE(ctx);

	// Anything the stub turned down goes to the interpreter, out of line
	e->deferred[e->nDeferred].site = slow;
	e->deferred[e->nDeferred].back = e->ptr;
	e->deferred[e->nDeferred].index = ctx->index;
	e->deferred[e->nDeferred].kind = 2;
	e->deferred[e->nDeferred].handler = handler;
	++e->nDeferred;
}

static bool _thumbEmitMultiple(struct ARMJitInsnContext* ctx, uint16_t opcode, uintptr_t handler) {
	if ((opcode & 0xFE00) == 0xB400) {
		// PUSH {rlist[, lr]}
		uint16_t mask = (opcode & 0xFF) | ((opcode & 0x100) ? (1 << ARM_LR) : 0);
		if (!mask) {
			return false;
		}
		_emitFastMultiple(ctx, false, ARM_SP, mask, 3, true, handler);
		return true;
	}
	if ((opcode & 0xFF00) == 0xBC00) {
		// POP {rlist} (without the PC)
		uint16_t mask = opcode & 0xFF;
		if (!mask) {
			return false;
		}
		_emitFastMultiple(ctx, true, ARM_SP, mask, 0, true, handler);
		return true;
	}
	if ((opcode & 0xF000) == 0xC000) {
		unsigned rn = (opcode >> 8) & 7;
		uint16_t mask = opcode & 0xFF;
		if (!mask) {
			return false;
		}
		if (opcode & 0x0800) {
			// LDMIA rn!, {rlist}: no writeback if rn is loaded
			_emitFastMultiple(ctx, true, rn, mask, 0, !(mask & (1 << rn)), handler);
		} else {
			_emitFastMultiple(ctx, false, rn, mask, 0, true, handler);
		}
		return true;
	}
	return false;
}

static bool _armEmitMultiple(struct ARMJitInsnContext* ctx, uint32_t opcode, uintptr_t handler) {
	if ((opcode >> 28) != COND_AL || (opcode & 0x0E000000) != 0x08000000) {
		return false;
	}
	uint16_t mask = opcode & 0xFFFF;
	unsigned rn = (opcode >> 16) & 0xF;
	bool load = opcode & 0x00100000;
	bool writeback = opcode & 0x00200000;
	if (!mask || (mask & 0x8000) || (opcode & 0x00400000) || rn == ARM_PC) {
		return false;
	}
	unsigned mode = ((opcode & 0x00800000) ? 0 : 2) | ((opcode & 0x01000000) ? 1 : 0);
	if (load && (mask & (1 << rn))) {
		writeback = false;
	}
	_emitFastMultiple(ctx, load, rn, mask, mode, writeback, handler);
	return true;
}

static bool _armEmitMemory(struct ARMJitInsnContext* ctx, uint32_t opcode, uintptr_t handler) {
	struct ARMJitEmitter* e = ctx->e;
	if ((opcode >> 28) != COND_AL || (opcode & 0x0C000000) != 0x04000000) {
		return false;
	}
	bool immediate = !(opcode & 0x02000000);
	bool pre = opcode & 0x01000000;
	bool up = opcode & 0x00800000;
	bool byte = opcode & 0x00400000;
	bool writeback = opcode & 0x00200000;
	bool load = opcode & 0x00100000;
	unsigned rn = (opcode >> 16) & 0xF;
	unsigned rd = (opcode >> 12) & 0xF;
	if (!immediate && (opcode & 0x10)) {
		return false; // Undefined encoding space
	}
	if (!pre && writeback) {
		return false; // LDRT/STRT
	}
	if (rd == ARM_PC) {
		return false;
	}
	bool doWriteback = !pre || writeback;
	if (doWriteback && (rn == ARM_PC || (load && rn == rd))) {
		return false;
	}
	if (!immediate) {
		unsigned rm = opcode & 0xF;
		unsigned shift = (opcode >> 5) & 3;
		unsigned shiftImm = (opcode >> 7) & 0x1F;
		if (rm == ARM_PC || (shift == SHIFT_ROR && !shiftImm)) {
			return false;
		}
	}

	struct ARMJitFastMem m = {
		.store = !load,
		.size = byte ? 1 : 4,
		.guestReg = rd,
		.writebackReg = -1
	};
	_loadGuestReg(ctx, R0, rn, 0);
	// r1 = offset
	if (immediate) {
		uint32_t offset = opcode & 0xFFF;
		if (pre) {
			_addImm(e, R0, R0, up ? (int32_t) offset : -(int32_t) offset);
			if (writeback) {
				m.writebackReg = rn;
				m.writebackDelta = 0;
			}
		} else {
			m.writebackReg = rn;
			m.writebackDelta = up ? (int32_t) offset : -(int32_t) offset;
		}
	} else {
		unsigned rm = opcode & 0xF;
		_loadGuestReg(ctx, R1, rm, 0);
		if (pre) {
			_dpReg(e, COND_AL, up ? ALU_ADD : ALU_SUB, false, R0, R0, R1, (opcode >> 5) & 3, (opcode >> 7) & 0x1F);
			if (writeback) {
				m.writebackReg = rn;
				m.writebackDelta = 0;
			}
		} else {
			// Post-indexed register offsets are rare; leave them to the interpreter
			return false;
		}
	}
	_emitMemoryWithFallback(ctx, &m, handler);
	return true;
}

/* ARM LDRH/LDRSH/LDRSB/STRH. Their handlers share LDR/STR's template, so the
 * same fast path applies with a different access size. */
static bool _armEmitMemoryHalf(struct ARMJitInsnContext* ctx, uint32_t opcode, uintptr_t handler) {
	struct ARMJitEmitter* e = ctx->e;
	if ((opcode >> 28) != COND_AL || (opcode & 0x0E000090) != 0x00000090 || !(opcode & 0x60)) {
		return false;
	}
	bool pre = opcode & 0x01000000;
	bool up = opcode & 0x00800000;
	bool immediate = opcode & 0x00400000;
	bool writeback = opcode & 0x00200000;
	bool load = opcode & 0x00100000;
	unsigned rn = (opcode >> 16) & 0xF;
	unsigned rd = (opcode >> 12) & 0xF;
	unsigned rm = opcode & 0xF;
	unsigned sh = (opcode >> 5) & 3;
	if (!load && sh != 1) {
		return false; // Not ARMv4
	}
	if (!pre && (writeback || !immediate)) {
		return false; // Post-indexed register offsets are rare
	}
	if (rd == ARM_PC || rn == ARM_PC || (!immediate && rm == ARM_PC)) {
		return false;
	}
	bool doWriteback = !pre || writeback;
	if (doWriteback && load && rn == rd) {
		return false;
	}
	struct ARMJitFastMem m = {
		.store = !load,
		.size = sh == 2 ? 1 : 2,
		.sign = sh != 1,
		.guestReg = rd,
		.writebackReg = -1
	};
	int32_t offset = (int32_t) (((opcode >> 4) & 0xF0) | (opcode & 0xF));
	if (!up) {
		offset = -offset;
	}
	_loadGuestReg(ctx, R0, rn, 0);
	if (!pre) {
		m.writebackReg = rn;
		m.writebackDelta = offset;
	} else {
		if (immediate) {
			_addImm(e, R0, R0, offset);
		} else {
			_loadGuestReg(ctx, R1, rm, 0);
			_dpReg(e, COND_AL, up ? ALU_ADD : ALU_SUB, false, R0, R0, R1, SHIFT_LSL, 0);
		}
		if (writeback) {
			m.writebackReg = rn;
			m.writebackDelta = 0;
		}
	}
	_emitMemoryWithFallback(ctx, &m, handler);
	return true;
}

/* ---------------------------------------------------------------------------
 * ARM translation
 * ------------------------------------------------------------------------- */

static bool _aluWritesRd(unsigned op) {
	return op < ALU_TST || op > ALU_CMN;
}

static bool _aluIsLogical(unsigned op) {
	switch (op) {
	case ALU_AND:
	case ALU_EOR:
	case ALU_TST:
	case ALU_TEQ:
	case ALU_ORR:
	case ALU_MOV:
	case ALU_BIC:
	case ALU_MVN:
		return true;
	default:
		return false;
	}
}

static bool _armIsNativeDataProcessing(uint32_t opcode) {
	if (opcode & 0x0C000000) {
		return false;
	}
	bool immediate = opcode & 0x02000000;
	if (!immediate && (opcode & 0x90) == 0x90) {
		return false; // Multiply, swap and halfword transfers
	}
	unsigned op = (opcode >> 21) & 0xF;
	bool s = opcode & 0x00100000;
	if (!_aluWritesRd(op) && !s) {
		return false; // MRS, MSR, BX, BKPT
	}
	unsigned rd = (opcode >> 12) & 0xF;
	if (rd == ARM_PC) {
		return false;
	}
	if (!immediate && (opcode & 0x10)) {
		unsigned rn = (opcode >> 16) & 0xF;
		unsigned rm = opcode & 0xF;
		unsigned rs = (opcode >> 8) & 0xF;
		if (rs == ARM_PC || rm == ARM_PC) {
			return false;
		}
		if (rn == ARM_PC && op != ALU_MOV && op != ALU_MVN) {
			return false;
		}
	}
	return true;
}

static bool _armIsBranchClass(uint32_t opcode) {
	if ((opcode & 0x0E000000) == 0x0A000000) {
		return true; // B, BL
	}
	if ((opcode & 0x0F000000) == 0x0F000000) {
		return true; // SWI
	}
	if ((opcode & 0x0E000000) == 0x0C000000 || (opcode & 0x0F000000) == 0x0E000000) {
		return true; // Coprocessor
	}
	if ((opcode & 0x0E000010) == 0x06000010) {
		return true; // Undefined
	}
	if ((opcode & 0x0D900000) == 0x01000000) {
		return true; // MRS/MSR/BX/BKPT/SWP space (cheap to be conservative here)
	}
	if ((opcode & 0x0E000000) == 0x08000000) {
		// LDM/STM: PC in list, empty list or S bit
		if ((opcode & 0x8000) || !(opcode & 0xFFFF) || (opcode & 0x00400000)) {
			return true;
		}
		if (((opcode >> 16) & 0xF) == ARM_PC) {
			return true;
		}
		return false;
	}
	if ((opcode & 0x0C000000) == 0x04000000 || (opcode & 0x0E000090) == 0x00000090) {
		// Single transfers: PC as destination or as written-back base
		if (((opcode >> 12) & 0xF) == ARM_PC || ((opcode >> 16) & 0xF) == ARM_PC) {
			return true;
		}
		return false;
	}
	if ((opcode & 0x0C000000) == 0) {
		// Data processing (incl. multiplies) writing the PC
		if (((opcode >> 12) & 0xF) == ARM_PC || ((opcode >> 16) & 0xF) == ARM_PC) {
			return true;
		}
	}
	return false;
}

static void _armEmitDataProcessing(struct ARMJitInsnContext* ctx, uint32_t opcode) {
	struct ARMJitEmitter* e = ctx->e;
	unsigned cond = opcode >> 28;
	unsigned op = (opcode >> 21) & 0xF;
	bool s = opcode & 0x00100000;
	bool immediate = opcode & 0x02000000;
	bool regShift = !immediate && (opcode & 0x10);
	unsigned rn = (opcode >> 16) & 0xF;
	unsigned rd = (opcode >> 12) & 0xF;
	uint32_t* skip = NULL;

	if (cond == COND_NV) {
		_nativeCycles(ctx);
		return;
	}

	bool readsFlags = cond != COND_AL;
	if (op == ALU_ADC || op == ALU_SBC || op == ALU_RSC) {
		readsFlags = true;
	}
	if (s && _aluIsLogical(op)) {
		readsFlags = true; // C may pass through, V always does
	}
	if (!immediate && !regShift && ((opcode >> 5) & 3) == SHIFT_ROR && !((opcode >> 7) & 0x1F)) {
		readsFlags = true; // RRX
	}

	if (readsFlags) {
		_flagsNeedHost(e);
	}
	if (cond != COND_AL) {
		// Both ways through a conditional instruction must agree on where the
		// flags are afterwards: in r8 and, when nothing disturbs them, the host
		_flagsToR8(e);
		skip = _bPlaceholder(e, cond ^ 1);
	}
	if (regShift) {
		_dpImm(e, COND_AL, ALU_ADD, false, R5, R5, 1);
	}

	if (op != ALU_MOV && op != ALU_MVN) {
		_loadGuestReg(ctx, R0, rn, regShift ? 4 : 0);
	}
	uint32_t operand2;
	if (immediate) {
		operand2 = opcode & 0xFFF;
	} else {
		unsigned rm = opcode & 0xF;
		_loadGuestReg(ctx, R1, rm, regShift ? 4 : 0);
		if (regShift) {
			unsigned rs = (opcode >> 8) & 0xF;
			_loadGuestReg(ctx, R2, rs, 0);
			operand2 = (R2 << 8) | (opcode & 0x70) | R1;
		} else {
			operand2 = (opcode & 0xFF0) | R1;
		}
	}

	/* The interpreter remembers the shifter carry-out of every data-processing
	 * instruction, and ARM MULS/MLAS later copy it into C. Keep it in step. */
	enum { CARRY_FROM_FLAGS_BEFORE, CARRY_CONSTANT, CARRY_FROM_RESULT_FLAGS, CARRY_COMPUTED } carryKind;
	uint32_t carryConstant = 0;
	if (immediate) {
		unsigned rot = (opcode >> 8) & 0xF;
		if (!rot) {
			carryKind = CARRY_FROM_FLAGS_BEFORE;
		} else {
			uint32_t imm = opcode & 0xFF;
			uint32_t value = (imm >> (2 * rot)) | (imm << (32 - 2 * rot));
			carryKind = CARRY_CONSTANT;
			carryConstant = value >> 31;
		}
	} else if (!regShift && !(opcode & 0xFE0)) {
		carryKind = CARRY_FROM_FLAGS_BEFORE; // LSL #0
	} else if (s && _aluIsLogical(op)) {
		carryKind = CARRY_FROM_RESULT_FLAGS;
	} else {
		carryKind = CARRY_COMPUTED;
	}
	switch (carryKind) {
	case CARRY_FROM_FLAGS_BEFORE:
		_flagsToR8(e);
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R8, SHIFT_ROR, 29);
		break;
	case CARRY_CONSTANT:
		_dpImm(e, COND_AL, ALU_MOV, false, R3, 0, carryConstant);
		break;
	case CARRY_COMPUTED:
		// A shift by zero passes the guest C through, so MOVS needs the flags
		_flagsNeedHost(e);
		_flagsToR8(e);
		// MOVS r3, operand2: host C becomes the shifter carry-out
		_emit(e, (COND_AL << 28) | (ALU_MOV << 21) | (1 << 20) | (R3 << 12) | operand2);
		_mrsFlags(e, R3);
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R3, SHIFT_ROR, 29);
		e->flags = FLAGS_R8;
		if (readsFlags) {
			_flagsNeedHost(e);
		}
		break;
	case CARRY_FROM_RESULT_FLAGS:
		break;
	}

	// Same instruction, host registers: rn = r0, rd = r0
	_emit(e, (COND_AL << 28) | (opcode & 0x03F00000) | (R0 << 16) | (R0 << 12) | operand2);
	if (_aluWritesRd(op)) {
		_storeGuestReg(ctx, R0, rd);
	}
	if (s) {
		_flagsWritten(e);
	}
	if (carryKind == CARRY_FROM_RESULT_FLAGS) {
		_flagsToR8(e);
		_dpReg(e, COND_AL, ALU_MOV, false, R3, 0, R8, SHIFT_ROR, 29);
	}
	_str(e, R3, R4, ctx->jit->offShifterCarry);
	if (skip) {
		// Skipped: r8 and the host flags both hold the flags. Match that if
		// possible, else fall back to r8 only.
		_flagsToR8(e);
		_patchBranch(skip, e->ptr);
	}
	_nativeCycles(ctx);
}

static void _armEmitInstruction(struct ARMJitInsnContext* ctx, uint32_t opcode, bool* endBlock) {
	if (_armIsNativeDataProcessing(opcode)) {
		_armEmitDataProcessing(ctx, opcode);
		return;
	}
	bool branchClass = _armIsBranchClass(opcode);
	ARMInstruction handler = _armTable[((opcode >> 16) & 0xFF0) | ((opcode >> 4) & 0x00F)];
	if (!branchClass && (_armEmitMemory(ctx, opcode, (uintptr_t) handler) || _armEmitMemoryHalf(ctx, opcode, (uintptr_t) handler) ||
	                     _armEmitMultiple(ctx, opcode, (uintptr_t) handler))) {
		return;
	}
	int chain = CHAIN_NONE;
	uint32_t target = 0;
	if ((opcode & 0x0E000000) == 0x0A000000) {
		// B/BL: static target
		int32_t offset = (int32_t) (opcode << 8) >> 6;
		uint32_t pcValue = ctx->block->guestAddress + (uint32_t) _pcValue(ctx);
		target = pcValue + (uint32_t) offset;
		chain = CHAIN_ALWAYS;
		unsigned cond = opcode >> 28;
		if (cond != COND_NV && _canBranchNatively(ctx, target)) {
			struct ARMJitEmitter* e = ctx->e;
			uint32_t link = (opcode & 0x01000000) ? pcValue - WORD_SIZE_ARM : 0;
			if (cond == COND_AL) {
				_emitNativeBranch(ctx, target, link);
				*endBlock = true;
				return;
			}
			_flagsNeedHost(e);
			_flagsToR8(e);
			uint32_t* skip = _bPlaceholder(e, cond ^ 1);
			int flags = e->flags;
			_emitNativeBranch(ctx, target, link);
			e->flags = flags;
			_patchBranch(skip, e->ptr);
			_cyclesAndCheck(ctx);
			return;
		}
	}
	_emitHelperCall(ctx, opcode >> 28, (uintptr_t) handler, branchClass, chain, target);
	if (branchClass && (opcode >> 28) == COND_AL) {
		*endBlock = true;
	}
}

/* ---------------------------------------------------------------------------
 * Thumb translation
 * ------------------------------------------------------------------------- */

static bool _thumbIsBranchClass(uint16_t opcode) {
	if ((opcode & 0xF000) == 0xD000) {
		return true; // Bcond, SWI, undefined
	}
	if ((opcode & 0xF800) == 0xE000 || (opcode & 0xF800) == 0xE800 || (opcode & 0xF800) == 0xF800) {
		return true; // B, BLX (undefined on ARMv4), BL suffix
	}
	if ((opcode & 0xFF00) == 0x4700) {
		return true; // BX
	}
	if ((opcode & 0xFF00) == 0xBD00 || (opcode & 0xFF00) == 0xBE00) {
		return true; // POP {pc}, BKPT
	}
	if ((opcode & 0xF800) == 0xC800 && !(opcode & 0xFF)) {
		return true; // LDMIA with empty list loads the PC
	}
	if ((opcode & 0xFC00) == 0x4400 && (opcode & 0x0087) == 0x0087 && (opcode & 0x0300) != 0x0100) {
		return true; // ADD/MOV with the PC as destination
	}
	if ((opcode & 0xF000) == 0xB000) {
		// Miscellaneous instructions not defined on ARMv4 trap
		if ((opcode & 0x0F00) != 0x0000 && (opcode & 0x0600) != 0x0400) {
			return true;
		}
	}
	return false;
}

static void _thumbEmitNative(struct ARMJitInsnContext* ctx, bool msrFirst, bool extraCycle) {
	if (msrFirst) {
		_flagsNeedHost(ctx->e);
	}
	if (extraCycle) {
		_dpImm(ctx->e, COND_AL, ALU_ADD, false, R5, R5, 1);
	}
}

/* Returns false if the instruction must be handled by the interpreter's handler */
static bool _thumbEmitNativeInstruction(struct ARMJitInsnContext* ctx, uint16_t opcode) {
	struct ARMJitEmitter* e = ctx->e;
	unsigned rd = opcode & 7;
	unsigned rs = (opcode >> 3) & 7;

	switch (opcode >> 13) {
	case 0: {
		unsigned op = (opcode >> 11) & 3;
		if (op != 3) {
			// LSL/LSR/ASR #imm5: MOVS rd, rm, shift #imm (same imm5 meaning as ARM)
			unsigned imm = (opcode >> 6) & 0x1F;
			_thumbEmitNative(ctx, true, false);
			_loadGuestReg(ctx, R1, rs, 0);
			_dpReg(e, COND_AL, ALU_MOV, true, R0, 0, R1, op, imm);
			_storeGuestReg(ctx, R0, rd);
			_flagsWritten(e);
			_nativeCycles(ctx);
			return true;
		}
		// ADD/SUB register or imm3
		bool imm3 = opcode & 0x0400;
		unsigned alu = (opcode & 0x0200) ? ALU_SUB : ALU_ADD;
		unsigned rn = (opcode >> 6) & 7;
		_loadGuestReg(ctx, R0, rs, 0);
		if (imm3) {
			_dpImm(e, COND_AL, alu, true, R0, R0, rn);
		} else {
			_loadGuestReg(ctx, R1, rn, 0);
			_dpReg(e, COND_AL, alu, true, R0, R0, R1, SHIFT_LSL, 0);
		}
		_storeGuestReg(ctx, R0, rd);
		_flagsWritten(e);
		_nativeCycles(ctx);
		return true;
	}
	case 1: {
		// MOV/CMP/ADD/SUB #imm8
		unsigned op = (opcode >> 11) & 3;
		unsigned reg = (opcode >> 8) & 7;
		uint32_t imm = opcode & 0xFF;
		switch (op) {
		case 0:
			_thumbEmitNative(ctx, true, false);
			_dpImm(e, COND_AL, ALU_MOV, true, R0, 0, imm);
			_storeGuestReg(ctx, R0, reg);
			break;
		case 1:
			_loadGuestReg(ctx, R0, reg, 0);
			_dpImm(e, COND_AL, ALU_CMP, true, 0, R0, imm);
			break;
		case 2:
			_loadGuestReg(ctx, R0, reg, 0);
			_dpImm(e, COND_AL, ALU_ADD, true, R0, R0, imm);
			_storeGuestReg(ctx, R0, reg);
			break;
		case 3:
			_loadGuestReg(ctx, R0, reg, 0);
			_dpImm(e, COND_AL, ALU_SUB, true, R0, R0, imm);
			_storeGuestReg(ctx, R0, reg);
			break;
		}
		_flagsWritten(e);
		_nativeCycles(ctx);
		return true;
	}
	case 2:
		if ((opcode & 0xFC00) == 0x4000) {
			// ALU operations
			unsigned op = (opcode >> 6) & 0xF;
			if (op == 0xD) {
				return false; // MUL: data-dependent timing
			}
			_loadGuestReg(ctx, R0, rd, 0);
			_loadGuestReg(ctx, R1, rs, 0);
			switch (op) {
			case 0x0: // AND
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_AND, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0x1: // EOR
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_EOR, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0x2: // LSL
			case 0x3: // LSR
			case 0x4: // ASR
			case 0x7: { // ROR
				unsigned shift = op == 0x2 ? SHIFT_LSL : op == 0x3 ? SHIFT_LSR : op == 0x4 ? SHIFT_ASR : SHIFT_ROR;
				_thumbEmitNative(ctx, true, true);
				_dpRegShiftReg(e, COND_AL, ALU_MOV, true, R0, 0, R0, shift, R1);
				_storeGuestReg(ctx, R0, rd);
				break;
			}
			case 0x5: // ADC
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_ADC, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0x6: // SBC
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_SBC, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0x8: // TST
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_TST, true, 0, R0, R1, SHIFT_LSL, 0);
				break;
			case 0x9: // NEG
				_dpImm(e, COND_AL, ALU_RSB, true, R0, R1, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0xA: // CMP
				_dpReg(e, COND_AL, ALU_CMP, true, 0, R0, R1, SHIFT_LSL, 0);
				break;
			case 0xB: // CMN
				_dpReg(e, COND_AL, ALU_CMN, true, 0, R0, R1, SHIFT_LSL, 0);
				break;
			case 0xC: // ORR
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_ORR, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0xE: // BIC
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_BIC, true, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			case 0xF: // MVN
				_thumbEmitNative(ctx, true, false);
				_dpReg(e, COND_AL, ALU_MVN, true, R0, 0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, rd);
				break;
			}
			_flagsWritten(e);
			_nativeCycles(ctx);
			return true;
		}
		if ((opcode & 0xFC00) == 0x4400) {
			// High register operations
			unsigned op = (opcode >> 8) & 3;
			unsigned hd = rd | ((opcode >> 4) & 8);
			unsigned hm = ((opcode >> 3) & 0xF);
			switch (op) {
			case 0: // ADD
				if (hd == ARM_PC) {
					return false;
				}
				_loadGuestReg(ctx, R0, hd, 0);
				_loadGuestReg(ctx, R1, hm, 0);
				_dpReg(e, COND_AL, ALU_ADD, false, R0, R0, R1, SHIFT_LSL, 0);
				_storeGuestReg(ctx, R0, hd);
				_nativeCycles(ctx);
				return true;
			case 1: // CMP
				_loadGuestReg(ctx, R0, hd, 0);
				_loadGuestReg(ctx, R1, hm, 0);
				_dpReg(e, COND_AL, ALU_CMP, true, 0, R0, R1, SHIFT_LSL, 0);
				_flagsWritten(e);
				_nativeCycles(ctx);
				return true;
			case 2: // MOV
				if (hd == ARM_PC) {
					return false;
				}
				_loadGuestReg(ctx, R1, hm, 0);
				_storeGuestReg(ctx, R1, hd);
				_nativeCycles(ctx);
				return true;
			default:
				return false; // BX
			}
		}
		return false;
	case 5:
		if ((opcode & 0xF000) == 0xA000) {
			unsigned reg = (opcode >> 8) & 7;
			uint32_t imm = (opcode & 0xFF) << 2;
			if (opcode & 0x0800) {
				// ADD rd, SP, #imm
				_loadGuestReg(ctx, R0, ARM_SP, 0);
				_addImm(e, R0, R0, (int32_t) imm);
			} else {
				// ADD rd, PC, #imm
				uint32_t value = ((ctx->block->guestAddress + (uint32_t) _pcValue(ctx)) & ~3u) + imm;
				_addImm(e, R0, R10, (int32_t) (value - ctx->block->guestAddress));
			}
			_storeGuestReg(ctx, R0, reg);
			_nativeCycles(ctx);
			return true;
		}
		if ((opcode & 0xFF00) == 0xB000) {
			// ADD/SUB SP, #imm7
			uint32_t imm = (opcode & 0x7F) << 2;
			_loadGuestReg(ctx, R0, ARM_SP, 0);
			_dpImm(e, COND_AL, (opcode & 0x80) ? ALU_SUB : ALU_ADD, false, R0, R0, imm);
			_storeGuestReg(ctx, R0, ARM_SP);
			_nativeCycles(ctx);
			return true;
		}
		return false;
	case 7:
		if ((opcode & 0xF800) == 0xF000) {
			// BL prefix: LR = PC + (offset << 12)
			int32_t offset = (int32_t) ((uint32_t) (opcode & 0x7FF) << 21) >> 9;
			uint32_t value = ctx->block->guestAddress + (uint32_t) _pcValue(ctx) + (uint32_t) offset;
			_addImm(e, R0, R10, (int32_t) (value - ctx->block->guestAddress));
			_storeGuestReg(ctx, R0, ARM_LR);
			_nativeCycles(ctx);
			return true;
		}
		return false;
	default:
		return false;
	}
}

static void _thumbEmitInstruction(struct ARMJitInsnContext* ctx, uint16_t opcode, bool* endBlock) {
	bool prevWasBL1 = ctx->prevWasBL1;
	ctx->prevWasBL1 = false;
	if ((opcode & 0xF800) == 0xF000) {
		ctx->prevWasBL1 = true;
		ctx->bl1Value = ctx->block->guestAddress + (uint32_t) _pcValue(ctx) + (uint32_t) ((int32_t) ((uint32_t) (opcode & 0x7FF) << 21) >> 9);
	}
	if (_thumbEmitNativeInstruction(ctx, opcode)) {
		return;
	}
	bool branchClass = _thumbIsBranchClass(opcode);
	ThumbInstruction handler = _thumbTable[opcode >> 6];
	if (!branchClass && (_thumbEmitMemory(ctx, opcode, (uintptr_t) handler) || _thumbEmitMultiple(ctx, opcode, (uintptr_t) handler))) {
		return;
	}
	uint32_t pcValue = ctx->block->guestAddress + (uint32_t) _pcValue(ctx);
	if ((opcode & 0xF000) == 0xD000 && (opcode & 0x0F00) < 0x0E00) {
		// Conditional branch: the handler checks the condition itself; when the
		// PC changed the branch was taken and we leave through a chain slot.
		uint32_t target = pcValue + (uint32_t) ((int32_t) (int8_t) opcode << 1);
		// Not taken (the common case in loops' exits and checks): the
		// interpreter only charges the prefetch, 1 + S cycles, which is r6
		struct ARMJitEmitter* e = ctx->e;
		if (e->nDeferred >= sizeof(e->deferred) / sizeof(e->deferred[0])) {
			e->overflow = true;
			return;
		}
		bool native = _canBranchNatively(ctx, target);
		_flagsNeedHost(e);
		_flagsToR8(e); // the taken path leaves the block
		if (native && target <= ctx->block->guestAddress + ctx->index * 2) {
			// Backward: most likely a loop, so keep the taken path in line
			uint32_t* notTaken = _bPlaceholder(e, ((opcode >> 8) & 0xF) ^ 1);
			int flags = e->flags;
			unsigned segSkip = e->segSkip;
			bool sawOther = e->sawOther;
			_emitRewind(e, segSkip);
			_emitNativeBranch(ctx, target, 0);
			e->flags = flags;
			e->segSkip = segSkip;
			e->sawOther = sawOther;
			_patchBranch(notTaken, e->ptr);
			_nativeCycles(ctx);
			return;
		}
		uint32_t* taken = _bPlaceholder(e, (opcode >> 8) & 0xF);
		e->deferred[e->nDeferred].rewind = e->segSkip;
		e->deferred[e->nDeferred].native = native;
		_nativeCycles(ctx);
		e->deferred[e->nDeferred].site = taken;
		e->deferred[e->nDeferred].back = e->ptr;
		e->deferred[e->nDeferred].index = ctx->index;
		e->deferred[e->nDeferred].kind = 0;
		e->deferred[e->nDeferred].handler = (uintptr_t) handler;
		e->deferred[e->nDeferred].target = target;
		++e->nDeferred;
		return;
	}
	int chain = CHAIN_NONE;
	uint32_t target = 0;
	uint32_t link = 0;
	if ((opcode & 0xF800) == 0xE000) {
		target = pcValue + (uint32_t) ((int32_t) ((uint32_t) (opcode & 0x7FF) << 21) >> 20);
		chain = CHAIN_ALWAYS;
	} else if ((opcode & 0xF800) == 0xF800 && prevWasBL1) {
		target = ctx->bl1Value + ((opcode & 0x7FF) << 1);
		chain = CHAIN_ALWAYS;
		link = pcValue - 1;
	}
	if (chain == CHAIN_ALWAYS && _canBranchNatively(ctx, target)) {
		_emitNativeBranch(ctx, target, link);
		*endBlock = true;
		return;
	}
	_emitHelperCall(ctx, COND_AL, (uintptr_t) handler, branchClass, chain, target);
	if (branchClass) {
		*endBlock = true;
	}
}

/* ---------------------------------------------------------------------------
 * Block compilation
 * ------------------------------------------------------------------------- */

static void _flushCache(struct ARMJit* jit, void* start, size_t size) {
	if (jit->flushCache) {
		jit->flushCache(start, size);
	} else {
		__builtin___clear_cache((char*) start, (char*) start + size);
	}
}

static bool _isJittableRegion(uint32_t address) {
	switch (address >> 24) {
	case GBA_REGION_BIOS:
	case GBA_REGION_EWRAM:
	case GBA_REGION_IWRAM:
	case GBA_REGION_ROM0:
	case GBA_REGION_ROM0_EX:
	case GBA_REGION_ROM1:
	case GBA_REGION_ROM1_EX:
	case GBA_REGION_ROM2:
	case GBA_REGION_ROM2_EX:
		return true;
	default:
		return false;
	}
}

static void _resetEmitter(struct ARMJitEmitter* e) {
	e->overflow = false;
	e->flags = FLAGS_R8;
	e->nConsts = 0;
	e->nConstUses = 0;
	e->nStubFixups = 0;
	e->nSlots = 0;
	e->nExits = 0;
	e->nBails = 0;
	e->nDeferred = 0;
	e->segSkip = 0;
	e->sawNative = false;
	e->sawOther = false;
}

/* Counts how many instructions from START on translate to native code with a
 * fixed cost (r6 plus at most one cycle), by translating them on a scratch
 * emitter. Such runs need only one event check up front (see _compile). */
static unsigned _scanNative(const struct ARMJitInsnContext* ctx, unsigned start, unsigned max, struct ARMJitEmitter* scratch, uint32_t* buffer, size_t words) {
	struct ARMJitInsnContext c = *ctx;
	c.e = scratch;
	unsigned n = 0;
	unsigned j;
	for (j = start; j < max && n < 64; ++j) {
		_resetEmitter(scratch);
		scratch->start = buffer;
		scratch->ptr = buffer;
		scratch->end = buffer + words;
		c.index = j;
		bool endBlock = false;
		if (c.thumb) {
			uint16_t opcode;
			LOAD_16(opcode, j * 2, c.block->hostCode);
			_thumbEmitInstruction(&c, opcode, &endBlock);
		} else {
			uint32_t opcode;
			LOAD_32(opcode, j * 4, c.block->hostCode);
			_armEmitInstruction(&c, opcode, &endBlock);
		}
		if (!scratch->sawNative || scratch->sawOther || endBlock || scratch->overflow) {
			break;
		}
		++n;
	}
	return n;
}

/* Returns how many instructions the block starting at ADDRESS should hold so
 * that no stretch of guest code is translated twice: it stops where another
 * block already starts, and where a branch from inside the block lands (loop
 * heads and if/else joins become blocks of their own). */
static unsigned _splitPoint(struct ARMJit* jit, const uint8_t* code, uint32_t address, bool thumb, unsigned max) {
	unsigned width = thumb ? 2 : 4;
	unsigned split = max;
	unsigned n;
	for (n = 0; n < max; ++n) {
		if (n && _lookup(jit, address + n * width, thumb)) {
			return n < split ? n : split;
		}
		uint32_t pc = address + n * width + 2 * width;
		uint32_t target = 0;
		bool hasTarget = false;
		bool ends = false;
		if (thumb) {
			uint16_t opcode;
			LOAD_16(opcode, n * 2, code);
			if ((opcode & 0xF000) == 0xD000 && (opcode & 0x0F00) < 0x0E00) {
				target = pc + (uint32_t) ((int32_t) (int8_t) opcode << 1);
				hasTarget = true;
			} else if ((opcode & 0xF800) == 0xE000) {
				target = pc + (uint32_t) ((int32_t) ((uint32_t) (opcode & 0x7FF) << 21) >> 20);
				hasTarget = true;
				ends = true;
			} else {
				ends = _thumbIsBranchClass(opcode);
			}
		} else {
			uint32_t opcode;
			LOAD_32(opcode, n * 4, code);
			if ((opcode & 0x0F000000) == 0x0A000000) {
				target = pc + (uint32_t) ((int32_t) (opcode << 8) >> 6);
				hasTarget = true;
			}
			ends = _armIsBranchClass(opcode) && (opcode >> 28) == COND_AL;
		}
		if (hasTarget && target > address && !((target - address) % width)) {
			unsigned index = (target - address) / width;
			if (index < split) {
				split = index;
			}
		}
		if (ends) {
			break;
		}
	}
	return split;
}

static struct ARMJitBlock* _compile(struct ARMJit* jit, struct ARMCore* cpu, uint32_t address, bool thumb) {
	struct GBA* gba = (struct GBA*) cpu->master;
	if (!_isJittableRegion(address) || gba->memory.activeRegion != (int) (address >> 24)) {
		return NULL;
	}
	uint32_t mask = cpu->memory.activeMask | (thumb ? 1 : 3);
	if (!cpu->memory.activeMask || !cpu->memory.activeRegion) {
		return NULL;
	}
	uint32_t offset = address & mask;
	uint32_t available = mask + 1 - offset;
	if (address >> 24 >= GBA_REGION_ROM0) {
		uint32_t romOffset = address & (GBA_SIZE_ROM0 - 1);
		if (romOffset >= gba->memory.romSize) {
			return NULL;
		}
		if (gba->memory.romSize - romOffset < available) {
			available = gba->memory.romSize - romOffset;
		}
	}
	unsigned width = thumb ? 2 : 4;
	unsigned maxInsns = thumb ? JIT_MAX_THUMB_INSNS : JIT_MAX_ARM_INSNS;
	// Prefetch syncing reads up to two instructions past the last one
	if (available < 3 * width) {
		return NULL;
	}
	if ((available - 2 * width) / width < maxInsns) {
		maxInsns = (available - 2 * width) / width;
	}
	if ((address & 0xFFFFFF) + (maxInsns + 2) * width > 0x1000000) {
		maxInsns = (0x1000000 - (address & 0xFFFFFF)) / width - 2;
	}

	if (jit->nBlocks >= JIT_MAX_BLOCKS || (size_t) (jit->coldStart - jit->codePtr) < JIT_MAX_BLOCK_WORDS ||
	    (size_t) (jit->codeEnd - jit->coldPtr) < JIT_MAX_BLOCK_WORDS) {
#ifdef JITDBG
		{ unsigned i, dead = 0; for (i = 0; i < jit->nBlocks; ++i) dead += jit->blocks[i].dead;
		_dbgDump();
		fprintf(stderr, "FULL blocks=%u dead=%u codeUsed=%u links=%u chain=%u\n", (unsigned) jit->nBlocks, dead, (unsigned) ((jit->codePtr - jit->codeStart) * 4), (unsigned) jit->nLinks, (unsigned) jit->nChainLinks); }
#endif
		_flush(jit, cpu);
	}

	struct ARMJitBlock* block = &jit->blocks[jit->nBlocks];
	memset(block, 0, sizeof(*block));
	block->guestAddress = address;
	block->thumb = thumb;
	block->ram = _ramOffset(address) >= 0;
	block->hostCode = (const uint8_t*) cpu->memory.activeRegion + offset;

	maxInsns = _splitPoint(jit, block->hostCode, address, thumb, maxInsns);

	struct ARMJitEmitter* e = jit->emitter;
retry:
	_resetEmitter(e);
	e->start = jit->codePtr;
	e->ptr = jit->codePtr;
	e->end = jit->codePtr + JIT_MAX_BLOCK_WORDS;
	e->otherPtr = jit->coldPtr;
	e->otherEnd = jit->coldPtr + JIT_MAX_BLOCK_WORDS;
	e->cold = false;

	/* Header: host code pointer and guest address. The full entry loads them
	 * and r6. A jump from a block in the same mapping and mode has r6 right
	 * already and moves r9 and r10 itself, so it skips to the fast entry. */
	_emit(e, (uint32_t) (uintptr_t) block->hostCode);
	_emit(e, address);
	block->entry = e->ptr;
	_emit(e, 0xE51F9010); // ldr r9, [pc, #-16]
	_emit(e, 0xE51FA010); // ldr r10, [pc, #-16]
	_ldr(e, R6, R4, thumb ? jit->offSeq16 : jit->offSeq32);
	_dpImm(e, COND_AL, ALU_ADD, false, R6, R6, 1);
	block->fastEntry = e->ptr;
	if (block->ram) {
		// Lets a store into this block's own code stop it (ARMJitInvalidateWrite)
		_loadConst(e, R0, (uint32_t) (uintptr_t) block);
		_str(e, R0, R4, jit->offCurrentBlock);
	}

	struct ARMJitInsnContext ctx = {
		.jit = jit,
		.e = e,
		.block = block,
		.gba = gba,
		.thumb = thumb
	};
	unsigned i;
	unsigned f;
	bool endBlock = false;
	struct ARMJitEmitter* scratch = jit->scratch;
	uint32_t* scratchBuffer = jit->scratchBuffer;
	for (i = 0; i < maxInsns && !endBlock && !e->overflow; ++i) {
		ctx.index = i;
		if (!e->segSkip) {
			/* A run of N native instructions: as long as the first N-1 can't
			 * reach the next event, only the last one needs an event check.
			 * Each costs at most r6 + 1, so check r5 + (N-1)(r6+1) < r7 once
			 * here; otherwise leave and let the interpreter step through the
			 * run, which is exactly what the per-instruction checks would do.
			 * Once the check passes, their r6 cycles are charged in one go. */
			unsigned run = _scanNative(&ctx, i, maxInsns, scratch, scratchBuffer, JIT_SCRATCH_WORDS);
			if (run >= 3 && e->nBails < sizeof(e->bails) / sizeof(e->bails[0])) {
				unsigned covered = run - 1;
				_flagsClobber(e);
				_dpImm(e, COND_AL, ALU_MOV, false, R3, 0, covered);
				_emit(e, 0xE0200090 | (R12 << 16) | (R5 << 12) | (R6 << 8) | R3); // mla r12, r3, r6, r5
				_dpImm(e, COND_AL, ALU_ADD, false, R12, R12, covered);
				_dpReg(e, COND_AL, ALU_CMP, true, 0, R12, R7, SHIFT_LSL, 0);
				e->bails[e->nBails].site = _bPlaceholder(e, COND_GE);
				e->bails[e->nBails].pcBefore = thumb ? (int32_t) (i * 2 + 2) : (int32_t) (i * 4 + 4);
				++e->nBails;
				e->segSkip = covered;
				_dpImm(e, COND_AL, ALU_SUB, false, R5, R12, covered);
			}
		}
		bool skipping = e->segSkip > 0;
#ifdef JITDBG
		uint32_t* dbgStart = e->ptr;
#endif
		if (thumb) {
			uint16_t opcode;
			LOAD_16(opcode, i * 2, block->hostCode);
			_thumbEmitInstruction(&ctx, opcode, &endBlock);
#ifdef JITDBG
			dbgThumbWords[opcode >> 8] += e->ptr - dbgStart; dbgThumbCount[opcode >> 8]++;
#endif
		} else {
			uint32_t opcode;
			LOAD_32(opcode, i * 4, block->hostCode);
			_armEmitInstruction(&ctx, opcode, &endBlock);
#ifdef JITDBG
			dbgArmWords[(opcode >> 20) & 0xFF] += e->ptr - dbgStart; dbgArmCount[(opcode >> 20) & 0xFF]++;
#endif
		}
		if (skipping && e->segSkip) {
			--e->segSkip;
		}
	}
	block->length = i;
	if (!endBlock && i) {
		// Fell off the end of the block: continue with the next block
		ctx.index = i - 1;
		_emitChainSlot(&ctx, address + i * (thumb ? 2 : 4), false, _pcValue(&ctx));
	}

	_emitConstPool(e);
#ifdef JITDBG
	uint32_t* dbgTail = e->ptr;
#endif
	// Out-of-line pieces
	_setCold(e, true);
	for (f = 0; f < e->nDeferred && !e->overflow; ++f) {
		ctx.index = e->deferred[f].index;
		_patchBranch(e->deferred[f].site, e->ptr);
		e->flags = FLAGS_R8;
		if (e->deferred[f].kind == 0) {
			// Taken conditional branch. Inside a run, the rest of the run was
			// charged already; take that back.
			_emitRewind(e, e->deferred[f].rewind);
			if (e->deferred[f].native) {
				_emitNativeBranch(&ctx, e->deferred[f].target, 0);
				continue;
			}
			_emitHelperCall(&ctx, COND_AL, e->deferred[f].handler, false, CHAIN_IF_TAKEN, e->deferred[f].target);
			_msrFlags(e, R8);
			_b(e, COND_AL, e->deferred[f].back);
			continue;
		}
		if (e->deferred[f].kind == 2) {
			// Multiple transfer the fast path turned down
			_emitHelperCall(&ctx, COND_AL, e->deferred[f].handler, false, CHAIN_NONE, 0);
			_b(e, COND_AL, e->deferred[f].back);
			continue;
		}
		// Memory access: r12 = 0 means done but an event is due, 1 means slow
		_dpImm(e, COND_AL, ALU_CMP, true, 0, R12, 0);
		uint32_t* slow = _bPlaceholder(e, COND_NE);
		if (!e->deferred[f].store) {
			_storeGuestReg(&ctx, R1, e->deferred[f].guestReg);
		}
		if (e->deferred[f].writebackReg >= 0) {
			_addImm(e, R0, R0, e->deferred[f].writebackDelta);
			_storeGuestReg(&ctx, R0, e->deferred[f].writebackReg);
		}
		if (e->nStubFixups < sizeof(e->stubFixups) / sizeof(e->stubFixups[0])) {
			e->stubFixups[e->nStubFixups].site = _bPlaceholder(e, COND_AL);
			e->stubFixups[e->nStubFixups].insn = ctx.index;
			++e->nStubFixups;
		} else {
			e->overflow = true;
		}
		_patchBranch(slow, e->ptr);
		_emitHelperCall(&ctx, COND_AL, e->deferred[f].handler, false, CHAIN_NONE, 0);
		_b(e, COND_AL, e->deferred[f].back);
	}

	// Chain slot exits: remember which slot we left through, then leave
	for (f = 0; f < e->nSlots; ++f) {
		uint32_t* stub = e->ptr;
		_loadConst(e, R1, (uint32_t) (uintptr_t) e->slots[f].site | (e->slots[f].fast ? 1 : 0));
		_str(e, R1, R4, jit->offLastSlot);
		_loadConst(e, R1, e->slots[f].target);
		_str(e, R1, R4, jit->offLastSlotTarget);
		if (e->slots[f].synced) {
			_b(e, COND_AL, jit->exitPlain);
		} else {
			_addImm(e, R0, R10, e->slots[f].pcAfter);
			_b(e, COND_AL, thumb ? jit->exitSyncThumb : jit->exitSyncArm);
		}
		_patchBranch(e->slots[f].site, stub);
	}

	// Event exit stubs
	uint32_t* stubs[JIT_MAX_ARM_INSNS + JIT_MAX_THUMB_INSNS] = {0};
	for (f = 0; f < e->nStubFixups; ++f) {
		unsigned insn = e->stubFixups[f].insn;
		if (!stubs[insn]) {
			stubs[insn] = e->ptr;
			int32_t pcAfter = thumb ? (int32_t) (insn * 2 + 4) : (int32_t) (insn * 4 + 8);
			_addImm(e, R0, R10, pcAfter);
			_b(e, COND_AL, thumb ? jit->exitSyncThumb : jit->exitSyncArm);
		}
		_patchBranch(e->stubFixups[f].site, stubs[insn]);
	}

	for (f = 0; f < e->nExits; ++f) {
		_patchBranch(e->exits[f].site, e->ptr);
		_addImm(e, R0, R10, e->exits[f].pcAfter);
		_b(e, COND_AL, thumb ? jit->exitSyncThumb : jit->exitSyncArm);
	}

	// Bails before a run that might reach the next event: step it in the interpreter
	for (f = 0; f < e->nBails; ++f) {
		_patchBranch(e->bails[f].site, e->ptr);
		_dpImm(e, COND_AL, ALU_MOV, false, R1, 0, 2);
		_str(e, R1, R4, jit->offExitRequest);
		_addImm(e, R0, R10, e->bails[f].pcBefore);
		_b(e, COND_AL, thumb ? jit->exitSyncThumb : jit->exitSyncArm);
	}

#ifdef JITDBG
	dbgTailWords += e->ptr - dbgTail;
#endif
	_setCold(e, false);

	if (e->overflow && i > 4) {
		// Too big (e.g. for the constant pool to stay in reach): try half
		maxInsns = i / 2;
		goto retry;
	}
	if (e->overflow || !i) {
		return NULL;
	}
	if (block->ram && !_registerPages(jit, block, block->length * width + 2 * width)) {
#ifdef JITDBG
		fprintf(stderr, "LINKS FULL\n");
#endif
		_flush(jit, cpu);
		return NULL;
	}

#ifdef JITLOG
	fprintf(stderr, "BLK %08X %d %u %u %u\n", address, thumb, i, (unsigned) (e->ptr - e->start), (unsigned) (e->otherPtr - jit->coldPtr));
#endif
	_flushCache(jit, e->start, (size_t) (e->ptr - e->start) * 4);
	_flushCache(jit, jit->coldPtr, (size_t) (e->otherPtr - jit->coldPtr) * 4);
	jit->codePtr = e->ptr;
	jit->coldPtr = e->otherPtr;

	unsigned h = _hash(address, thumb);
	block->nextInHash = jit->hash[h];
	jit->hash[h] = block;
	++jit->nBlocks;
	++jit->stats.blocksCompiled;
	return block;
}

/* ---------------------------------------------------------------------------
 * Public interface
 * ------------------------------------------------------------------------- */

static void _freeJit(struct ARMJit* jit) {
	free(jit->blocks);
	free(jit->links);
	free(jit->chainLinks);
	free(jit->emitter);
	free(jit->scratch);
	free(jit->scratchBuffer);
	free(jit);
}

bool ARMJitInit(struct ARMCore* cpu, void* execMemory, size_t size, ARMJitCacheFlush flush) {
	if (cpu->jit) {
		ARMJitDeinit(cpu);
	}
	if (!execMemory || size < (JIT_TRAMPOLINE_WORDS + 6 * JIT_MAX_BLOCK_WORDS) * 4) {
		return false;
	}
	struct ARMJit* jit = calloc(1, sizeof(*jit));
	if (!jit) {
		return false;
	}
	jit->blocks = calloc(JIT_MAX_BLOCKS, sizeof(*jit->blocks));
	jit->links = calloc(JIT_MAX_LINKS, sizeof(*jit->links));
	jit->chainLinks = calloc(JIT_MAX_CHAIN_LINKS, sizeof(*jit->chainLinks));
	jit->emitter = malloc(sizeof(*jit->emitter));
	jit->scratch = malloc(sizeof(*jit->scratch));
	jit->scratchBuffer = malloc(JIT_SCRATCH_WORDS * 4);
	if (!jit->blocks || !jit->links || !jit->chainLinks || !jit->emitter || !jit->scratch || !jit->scratchBuffer) {
		_freeJit(jit);
		return false;
	}
#ifdef JITDBG
	atexit(_dbgDump);
#endif
	jit->mem = execMemory;
	jit->memWords = size / 4;
	jit->flushCache = flush;
	jit->offCycles = offsetof(struct ARMCore, cycles);
	jit->offNextEvent = offsetof(struct ARMCore, nextEvent);
	jit->offCpsr = offsetof(struct ARMCore, cpsr);
	jit->offPrefetch0 = offsetof(struct ARMCore, prefetch);
	jit->offPrefetch1 = offsetof(struct ARMCore, prefetch) + 4;
	jit->offSeq32 = offsetof(struct ARMCore, memory.activeSeqCycles32);
	jit->offSeq16 = offsetof(struct ARMCore, memory.activeSeqCycles16);
	jit->offExitRequest = offsetof(struct ARMCore, jitExitRequest);
	jit->offCurrentBlock = offsetof(struct ARMCore, jitCurrentBlock);
	jit->offLastSlot = offsetof(struct ARMCore, jitLastSlot);
	jit->offLastSlotTarget = offsetof(struct ARMCore, jitLastSlotTarget);
	jit->offShifterCarry = offsetof(struct ARMCore, shifterCarryOut);

	if (!cpu->master || !_buildTrampolines(jit, (struct GBA*) cpu->master)) {
		_freeJit(jit);
		return false;
	}
	if (flush) {
		flush(jit->mem, JIT_TRAMPOLINE_WORDS * 4);
	} else {
		__builtin___clear_cache((char*) jit->mem, (char*) (jit->mem + JIT_TRAMPOLINE_WORDS));
	}
	jit->codeStart = jit->mem + JIT_TRAMPOLINE_WORDS;
	jit->codePtr = jit->codeStart;
	jit->codeEnd = jit->mem + jit->memWords;
	jit->coldStart = jit->codeStart + (jit->codeEnd - jit->codeStart) * JIT_HOT_PERCENT / 100;
	jit->coldPtr = jit->coldStart;
	jit->enabled = true;

	cpu->jit = jit;
	cpu->jitCodePages = jit->pages;
	cpu->jitExitRequest = 0;
	return true;
}

void ARMJitDeinit(struct ARMCore* cpu) {
	struct ARMJit* jit = cpu->jit;
	if (!jit) {
		return;
	}
	cpu->jit = NULL;
	cpu->jitCodePages = NULL;
	_freeJit(jit);
}

void ARMJitSetEnabled(struct ARMCore* cpu, bool enabled) {
	if (cpu->jit) {
		cpu->jit->enabled = enabled;
		_flush(cpu->jit, cpu);
		cpu->jitCodePages = enabled ? cpu->jit->pages : NULL;
	}
}

bool ARMJitIsEnabled(struct ARMCore* cpu) {
	return cpu->jit && cpu->jit->enabled;
}

void ARMJitFlush(struct ARMCore* cpu) {
	if (cpu->jit) {
		_flush(cpu->jit, cpu);
	}
}

static inline unsigned _fastIndex(uint32_t key) {
	return (key >> 1) & (JIT_FAST_SIZE - 1);
}

static void _unchain(struct ARMJit* jit, struct ARMJitBlock* block) {
	unsigned index = _fastIndex(block->guestAddress | block->thumb);
	if (jit->fast[index].entry == block->entry) {
		jit->fast[index].key = 0;
		jit->fast[index].entry = NULL;
	}
	struct ARMJitChainLink* link = block->incoming;
	while (link) {
		*link->slot = link->original;
		_flushCache(jit, link->slot, 4);
		link = link->next;
	}
	block->incoming = NULL;
}

static void _chain(struct ARMJit* jit, uint32_t* taggedSlot, struct ARMJitBlock* target) {
	if (jit->nChainLinks >= JIT_MAX_CHAIN_LINKS) {
		return;
	}
	// Bit 0 of the slot address: r9/r10 are already set for the target
	bool fast = (uintptr_t) taggedSlot & 1;
	uint32_t* slot = (uint32_t*) ((uintptr_t) taggedSlot & ~(uintptr_t) 3);
	struct ARMJitChainLink* link = &jit->chainLinks[jit->nChainLinks++];
	link->slot = slot;
	link->original = *slot;
	link->next = target->incoming;
	target->incoming = link;
	*slot = _branchWord(COND_AL, slot, fast ? target->fastEntry : target->entry);
	_flushCache(jit, slot, 4);
}

#ifdef JITDBG
unsigned long long dbgInval[ARM_JIT_RAM_PAGES];
void _dbgInvalDump(void) {
	unsigned i;
	for (i = 0; i < ARM_JIT_RAM_PAGES; ++i) if (dbgInval[i] > 20) {
		unsigned off = i << ARM_JIT_PAGE_SHIFT;
		fprintf(stderr, "INVAL %08X x%llu\n", off >= ARM_JIT_EWRAM_SIZE ? 0x03000000 + off - ARM_JIT_EWRAM_SIZE : 0x02000000 + off, dbgInval[i]);
	}
}
#endif
void ARMJitInvalidateWrite(struct ARMCore* cpu, unsigned offset) {
	struct ARMJit* jit = cpu->jit;
	unsigned page = offset >> ARM_JIT_PAGE_SHIFT;
	if (!jit || page >= ARM_JIT_RAM_PAGES) {
		return;
	}
	/* Only blocks whose guest code overlaps the written word are stale. Games
	 * often keep variables right next to code in RAM (Metroid Fusion has a
	 * frame counter just before its interrupt handler), and throwing away
	 * every block in the page on such writes means recompiling every frame. */
	uint32_t lo = offset & ~3u;
	uint32_t hi = lo + 4;
	struct ARMJitPageLink** linkp = &jit->pageHead[page];
	bool live = false;
	while (*linkp) {
		struct ARMJitPageLink* link = *linkp;
		struct ARMJitBlock* block = link->block;
		if (!block->dead && block->ramStart < hi && lo < block->ramEnd) {
			block->dead = 1;
			_unlinkFromHash(jit, block);
			_unchain(jit, block);
			if (block == cpu->jitCurrentBlock) {
				cpu->jitExitRequest = 1;
			}
			++jit->stats.invalidations;
#ifdef JITDBG
			++dbgInval[page];
#endif
		}
		if (block->dead) {
			*linkp = link->next; // drop links to dead blocks
			continue;
		}
		live = true;
		linkp = &link->next;
	}
	jit->pages[page] = live;
}

const struct ARMJitStats* ARMJitGetStats(struct ARMCore* cpu) {
	struct ARMJit* jit = cpu->jit;
	if (!jit) {
		return NULL;
	}
	jit->stats.hotBytes = (uint32_t) ((jit->codePtr - jit->codeStart) * 4);
	jit->stats.coldBytes = (uint32_t) ((jit->coldPtr - jit->coldStart) * 4);
	return &jit->stats;
}

static bool _prefetchMatches(const struct ARMCore* cpu, const struct ARMJitBlock* block) {
	if (block->thumb) {
		uint16_t a, b;
		LOAD_16(a, 0, block->hostCode);
		LOAD_16(b, 2, block->hostCode);
		return cpu->prefetch[0] == a && cpu->prefetch[1] == b;
	}
	uint32_t a, b;
	LOAD_32(a, 0, block->hostCode);
	LOAD_32(b, 4, block->hostCode);
	return cpu->prefetch[0] == a && cpu->prefetch[1] == b;
}

void ARMJitRunLoop(struct ARMCore* cpu) {
	struct ARMJit* jit = cpu->jit;
	struct GBA* gba = (struct GBA*) cpu->master;
	if (jit->idleOptimization != (int) gba->idleOptimization || jit->idleLoop != gba->idleLoop) {
		// Blocks were translated for the old idle loop settings
		jit->idleOptimization = gba->idleOptimization;
		jit->idleLoop = gba->idleLoop;
		_flush(jit, cpu);
	}
	while (cpu->cycles < cpu->nextEvent) {
		bool thumb = cpu->executionMode == MODE_THUMB;
		uint32_t address = cpu->gprs[ARM_PC] - (thumb ? WORD_SIZE_THUMB : WORD_SIZE_ARM);
		struct ARMJitBlock* block = _lookup(jit, address, thumb);
		if (!block && !jit->stepping) {
			block = _compile(jit, cpu, address, thumb);
		}
		if (!block) {
			if (!jit->stepping) {
				++jit->stats.compileFailures;
				jit->stats.lastFailAddress = address | thumb;
			}
		} else if (!_prefetchMatches(cpu, block)) {
			++jit->stats.prefetchMismatches;
			jit->stats.lastFailAddress = address | thumb;
			block = NULL;
		}
		if (!block || jit->stepping) {
			/* After a bail, take at least one interpreter step, then keep
			 * stepping until an existing block's entry turns up, so that the
			 * event lands on exactly the same instruction as in the
			 * interpreter without compiling blocks at odd entry points */
			if (jit->stepping && block && jit->stepping < JIT_STEP_LIMIT) {
				jit->stepping = 0;
			} else {
				if (jit->stepping) {
					--jit->stepping;
				}
				cpu->jitLastSlot = NULL;
				ARMStepOne(cpu);
				++jit->stats.interpretedSteps;
				continue;
			}
		}
		if (cpu->jitLastSlot) {
			if (cpu->jitLastSlotTarget == (block->guestAddress | block->thumb)) {
				_chain(jit, cpu->jitLastSlot, block);
			}
			cpu->jitLastSlot = NULL;
		}
		unsigned index = _fastIndex(block->guestAddress | block->thumb);
		jit->fast[index].key = block->guestAddress | block->thumb;
		jit->fast[index].entry = block->entry;
		cpu->jitExitRequest = 0;
		jit->enter(cpu, block->entry);
		cpu->jitCurrentBlock = NULL;
		++jit->stats.blocksEntered;
		if (cpu->jitExitRequest == 2) {
			cpu->jitExitRequest = 0;
			jit->stepping = JIT_STEP_LIMIT;
		}
	}
	cpu->irqh.processEvents(cpu);
}

#else

bool ARMJitInit(struct ARMCore* cpu, void* execMemory, size_t size, ARMJitCacheFlush flush) {
	UNUSED(cpu);
	UNUSED(execMemory);
	UNUSED(size);
	UNUSED(flush);
	return false;
}

void ARMJitDeinit(struct ARMCore* cpu) {
	UNUSED(cpu);
}

void ARMJitSetEnabled(struct ARMCore* cpu, bool enabled) {
	UNUSED(cpu);
	UNUSED(enabled);
}

bool ARMJitIsEnabled(struct ARMCore* cpu) {
	UNUSED(cpu);
	return false;
}

void ARMJitFlush(struct ARMCore* cpu) {
	UNUSED(cpu);
}

void ARMJitInvalidateWrite(struct ARMCore* cpu, unsigned offset) {
	UNUSED(cpu);
	UNUSED(offset);
}

void ARMJitRunLoop(struct ARMCore* cpu) {
	UNUSED(cpu);
}

const struct ARMJitStats* ARMJitGetStats(struct ARMCore* cpu) {
	UNUSED(cpu);
	return NULL;
}

#endif
