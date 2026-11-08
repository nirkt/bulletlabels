/* Special-bar setup of the simulation's characters (bullet-label round 2, 2026-10-15).
 *
 * Root cause of the Gus / Mortis / Bibi / ... crashes at tick 3: the client sets up a character's special bar from its
 * character data when it first decodes it (0xb2fe44 -> 0xb3020c: UniqueProperty 1 / 3 / 10 / 12 / 17, the skill at
 * data +0x260 with flag +0x1b4, the data at +0x268 of kind 15). The bar type selects how many bits the snapshot carries
 * for the bar value (tag 0xb4: decoder 0xb2de2c switch on client +0x308, encoder 0xb360ac switch on simulation +0xc08).
 * The simulation's character init in this libg (0xb34e3c) sets the bar type only from cards (0x74..0x77); the
 * data-driven part is not in the client build. So the simulation writes no bar field (type 0) while the client reads
 * one, every later field of the snapshot is misread, and the client builds garbage objects (null element of the
 * card list read by 0xba8374, players[] entry at 0xb2de98, ...).
 *
 * Fix: before the simulation's character init, apply the same data-driven rules as the client's 0xb3020c (including
 * its final reset to no bar when 0xab41dc(data) holds, 0xb30680), with the
 * client fields mapped to the simulation's (client +0x308 type -> +0xc08, +0x30c max -> +0xc0c, +0x304 value -> +0xbf8:
 * the decoder stores the 0xb4 value at +0x304, the encoder writes +0xbf8, the bar logic 0xb420ac fills +0xbf8 up to
 * +0xc0c). The card-driven overrides (0x74..0x77) then still come from the init, as on the client (cards after data).
 * The player-card cases of 0xb3020c (player card lookups 0x3b, 0x41, 0x42, 0x51, 0x79, 0x80, 0x82, 0x83, 0x64, 0xc3,
 * 0xb5 and the player's gadget at +400) are not mirrored: the capture environment gives players no cards. */
#include "nbhost.h"
#include <stdint.h>
extern Module *g_libg;
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
typedef void *(*fp_p)(void *);
typedef int (*fi_p)(void *);
static long g_bars;

void bar_fix_setup(uint8_t *ch) {
    void *d = ((fp_p)B(0xb5d7a4))(ch); if (!d) return;            /* the character's data (as the client's 0xb3020c) */
    int up = ((fi_p)B(0xab3d04))(d);                                /* UniqueProperty (data +0x294) */
    int type = I32(ch, 0xc08), max = I32(ch, 0xc0c), val = I32(ch, 0xbf8), set = 0;
    void *sk = PTR(d, 0x260);
    /* the skill at data +0x260 with flag +0x1b4: the client gives type 0xb with UniqueProperty 0x45 (Bibi), else 1.
     * Only the 0x45 case is mirrored: with type 1 (Gladiator, the only playable brawler that reaches it) the client
     * then faults at tick 3 in its skill decoder (pc 0xb0def8, lr 0xb7a298), while without it Gladiator runs with the
     * client in sync (positions track the simulation). The fault is a client skill object with a null data pointer
     * (0xb7a294 -> 0xb0def8(skill +0)), reached when the bar is live: the environment's skill-slot setup, not the bar
     * (loader_fix/APPLY.md). */
    if (sk && (((fi_p)B(0xb0e170))(sk) & 1) && up == 0x45) { type = 0xb; max = ((fi_p)B(0xb0e178))(sk); set = 1; }
    if (up == 3) { type = 0xb; max = ((fi_p)B(0xab404c))(d) * 0x32; set = 1; }
    if (up == 1) { val = 0; type = 8; max = 1000; set = 1; }
    void *k = PTR(d, 0x268);
    if (k && ((fi_p)B(0xb0dc34))(k) == 0xf) {
        type = 7; val = 0; set = 1;
        max = ((int (*)(void *, void *))B(0xbb325c))(((fp_p)B(0xb0d910))(k), ch + 0x48);   /* the simulation's card list */
    }
    if (up == 10) { type = 9; val = 0; max = ((fi_p)B(0xab404c))(d); set = 1; }
    if (up == 0xc) { type = 10; val = 0; max = ((fi_p)B(0xab404c))(d); set = 1; }
    if (up == 0x11) { type = 6; val = 0; max = ((fi_p)B(0xab404c))(d); set = 1; }
    if (!set) return;
    /* the client's last step (0xb30678..0xb30680, after the card overrides 0x74..0x76): no bar when the data's list at
     * +0x2b0 holds an entry of kind 0x16 (0xab41dc); e.g. Gladiator, whose skill would give type 1 */
    if (((fi_p)B(0xab41dc))(d) & 1) type = 0;
    I32(ch, 0xc08) = type; I32(ch, 0xc0c) = max; I32(ch, 0xbf8) = val;
    if (g_bars++ < 20) LOG("[bar-fix] character %p UniqueProperty %d: bar type %d max %d value %d\n", ch, up, type, max, val);
}
