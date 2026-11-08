/* test-only: NB_TRACE=hexoff,hexoff,... logs every entry of those libg functions (args, caller, sim tick) */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
extern Module *g_libg; extern void *g_sim;
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx));
static uint64_t t_addr[64]; static int t_n;
static long t_cnt[64];
static void tr(int i, uint64_t *c) {
    int tick = g_sim ? *(int32_t *)((uint8_t *)g_sim + 0x10) : -1;
    if (t_cnt[i]++ >= 4000) return;
    uint64_t vt = 0; if (c[0] > 0x10000 && (c[0] & 7) == 0) { uint64_t v = *(uint64_t *)c[0]; vt = v - (uint64_t)g_libg->base; if (vt > 0x2000000) vt = 0; }
    int id4 = -1; { uint64_t p = *(uint64_t *)((uint8_t *)c + 160 + 0); (void)p; }
    { uint64_t x4 = c[4]; (void)x4; }
    if (t_addr[i] == 0xb28f84 || t_addr[i] == 0xb29494 || t_addr[i] == 0xb28c0c || t_addr[i] == 0xb28d4c) {
        uint8_t *o = (uint8_t *)c[0], *d = *(uint8_t **)(o + 0x10); int id = -1, ty = -1;
        if (d) { id = ((int (*)(void *))((uint8_t *)g_libg->base + 0xac3a80))(*(void **)(d + 0x58)); ty = *(int32_t *)(d + 0x118); }
        LOG("[trace-area] %lx tick %d obj=%p data=%p id=%d type=%d cb=%ld +0x330=%p +0x338=%p f4=%d\n", (unsigned long)t_addr[i], tick, o, d, id, ty,
            (long)((*(uint8_t **)(o + 0xc0) - *(uint8_t **)(o + 0xb8)) / 0x40), *(void **)(o + 0x330), *(void **)(o + 0x338), o[0xf4]);
    }
    uint64_t vt1 = 0; if (c[1] > 0x10000 && (c[1] & 7) == 0) { uint64_t v = *(uint64_t *)c[1]; vt1 = v - (uint64_t)g_libg->base; if (vt1 > 0x2000000) vt1 = 0; }
    LOG("[trace] vt1=%lx ", (unsigned long)vt1);
    LOG("[trace] %lx tick %d x0=%lx x1=%lx x2=%lx x3=%lx x4=%lx lr=%lx vt0=%lx\n", (unsigned long)t_addr[i], tick, c[0], c[1], c[2], c[3], c[4], (unsigned long)(c[9] - (uint64_t)g_libg->base), (unsigned long)vt);
}
#define CB(n) static void cb##n(uint64_t *c) { tr(n, c); }
CB(0) CB(1) CB(2) CB(3) CB(4) CB(5) CB(6) CB(7) CB(8) CB(9) CB(10) CB(11) CB(12) CB(13) CB(14) CB(15) CB(16) CB(17) CB(18) CB(19) CB(20) CB(21) CB(22) CB(23) CB(24) CB(25) CB(26) CB(27) CB(28) CB(29) CB(30) CB(31)
static void (*cbs[32])(uint64_t *) = {cb0,cb1,cb2,cb3,cb4,cb5,cb6,cb7,cb8,cb9,cb10,cb11,cb12,cb13,cb14,cb15,cb16,cb17,cb18,cb19,cb20,cb21,cb22,cb23,cb24,cb25,cb26,cb27,cb28,cb29,cb30,cb31};
void trace_install(void) {
    const char *e = getenv("NB_TRACE"); if (!e) return;
    char *s = strdup(e), *tok = strtok(s, ",");
    while (tok && t_n < 32) { t_addr[t_n] = strtoull(tok, NULL, 16); hook_enter(g_libg, t_addr[t_n], cbs[t_n]); LOG("[trace] hook %lx\n", (unsigned long)t_addr[t_n]); t_n++; tok = strtok(NULL, ","); }
}
