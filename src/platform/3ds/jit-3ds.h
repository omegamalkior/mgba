/* 3DS support for the experimental ARM dynarec.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef CTR_JIT_H
#define CTR_JIT_H

#include <mgba-util/common.h>

struct ARMCore;

/* Attaches the dynarec to the CPU (once) and turns it on or off. Returns true
 * if the dynarec is running afterwards. */
bool ctrJitAttach(struct ARMCore* cpu, bool enabled);
const char* ctrJitStatus(void);

#endif
