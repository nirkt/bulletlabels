/* spawn_fix (2026-11-04): the child objects that the shipped client libg does NOT create.
 *
 * Evidence (see the report): libg v68.279 is the client build of the shared logic; the server-only code that creates the children
 * of an explosion is compiled out. Examples: the area apply 0xb29494 returns early for every area type outside the mask
 * 0x10400440000001 (Damage, ...), BulletExplosion (type 5) is only read at 0xb76e80 (z of the area), the projectile end 0xb75998
 * calls the ChainBullet getter 0xafa920 at 0xb76118 and throws the result away. The offline battle therefore never has the
 * server's part; here it is re-implemented with libg's own spawn function.
 *
 * BulletExplosion (area data +0x118 == 5): N = CustomValue (+0x13c) copies of BulletExplosionBullet (+0x78) leave the area's
 * position evenly spread over 360 degrees, starting at the angle of the parent's flight (area +0x140) plus BulletExplosionBulletOffset
 * (+0x148 is not an angle in the data: unused here), each flying BulletExplosionBulletDistance (+0x144) * 100 world units. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
extern Module *g_libg; extern void *g_sim;
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx));
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
static int mode(void) { return I32(g_libg->base, 0x12eefb4) == 3; }
int g_spawn_fix = 1, g_spawn_dbg = 0;
static long g_nbe, g_nbe_proj;

typedef long (*spawn_t)(int, int, void *, void *, void *, int, int, int, int, int, int, void *, void *, int, int, void *);
typedef int (*i1_t)(int);
typedef int (*i2_t)(int, int);
static void *g_cur_proj; static int g_cur_tick = -1;
static int rec;
extern void *g_client_obj;
int g_spawn_chain = 1;
static long g_nchain, g_nchain_proj, g_nchain_skip[4];
static int place_one_force(uint8_t *tm, void *data, int x, int y) {
    if (!tm || !data) return 0;
    int w = I32(tm, 0xc4), h = I32(tm, 0xc8);
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return 0;
    uint8_t **cells = (uint8_t **)PTR(tm, 0x20); if (!cells || !cells[y * w + x]) return 0;
    ((void (*)(void *, void *, int, int, int))B(0xa52df0))(tm, data, x, y, 0);
    return 1;
}

/* ---- WallSpawn (area type 15): dynamic wall tiles. The server-side tile setter is present but unreferenced in libg: 0xa52df0
 * (tilemap, tileData, x, y, value): cell.setData + dynamic list + dirty flag. Tile data by DynamicCode: 0xa50dec. The sim's tile
 * map is *(sim+0x68); the client keeps a copy at *(client+0xf8) (filled from the sim's snapshot; the client listener is a stub
 * here), so both are set. */
int g_spawn_walls = 1;
static int tile_id(void *d);
static void *g_mapview;
static void on_mapview(uint64_t *c) { g_mapview = (void *)c[0]; }
static long g_nwall_areas, g_nwall_tiles; static int g_last_ring[2] = {-1, -1};
static int cell_id(uint8_t *tm, int x, int y) { if (!tm) return -1; uint8_t **c = (uint8_t **)PTR(tm, 0x20); uint8_t *ce = c ? c[y * I32(tm, 0xc4) + x] : NULL; return ce ? tile_id(PTR(ce, 0)) : -2; }
void spawnfix_walldiag(const char *why) { if (g_last_ring[0] < 0 || !g_spawn_dbg) return; uint8_t *ts = g_sim ? PTR(g_sim, 0x68) : NULL, *tc = g_client_obj ? PTR(g_client_obj, 0xf8) : NULL; LOG("[spawnfix] walldiag %s tick %d: ring tile (%d,%d) sim %d client %d; (%d,%d) sim %d client %d dynlist sim %d client %d\\n", why, g_sim ? I32(g_sim, 0x10) : -1, g_last_ring[0] + 2, g_last_ring[1], cell_id(ts, g_last_ring[0] + 2, g_last_ring[1]), cell_id(tc, g_last_ring[0] + 2, g_last_ring[1]), g_last_ring[0] + 2, g_last_ring[1] + 1, cell_id(ts, g_last_ring[0] + 2, g_last_ring[1] + 1), cell_id(tc, g_last_ring[0] + 2, g_last_ring[1] + 1), ts ? I32(ts, 0x34) : -1, tc ? I32(tc, 0x34) : -1); }
static void *tile_by_code(int code) { return ((void *(*)(int))B(0xa50dec))(code); }
static int tile_id(void *d) { return d ? I32(d, 0x20) : 0; }
static int place_one(uint8_t *tm, void *data, int x, int y) {
    if (!tm || !data) return 0;
    int w = I32(tm, 0xc4), h = I32(tm, 0xc8);
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return 0;
    uint8_t **cells = (uint8_t **)PTR(tm, 0x20); if (!cells) return 0;
    uint8_t *cell = cells[y * w + x]; if (!cell) return 0;
    void *cur = PTR(cell, 0);
    if (tile_id(cur) != 27000000) return 0;     /* only open ground */
    ((void (*)(void *, void *, int, int, int))B(0xa52df0))(tm, data, x, y, 0);
    return 1;
}
static const struct { int skin_proj, code; } g_wall_codes[] = {
#include "wall_codes.h"
    {0, 0}};
static int wall_code_for(uint8_t *pr) {
    if (!pr) return 1;
    uint8_t *d = PTR(pr, 0x10); if (!d || !PTR(d, 0x58)) return 1;
    int id = ((int (*)(void *))B(0xac3a80))(PTR(d, 0x58));
    for (int i = 0; g_wall_codes[i].skin_proj; i++) if (g_wall_codes[i].skin_proj == id) return g_wall_codes[i].code;
    return 1;
}
static void wall_ring(uint8_t *a, uint8_t *d) {
    uint8_t *pr = (g_cur_proj && g_cur_tick == (g_sim ? I32(g_sim, 0x10) : -1)) ? (uint8_t *)g_cur_proj : NULL;
    int code = wall_code_for(pr);
    void *wall = tile_by_code(code), *fill = tile_by_code(code + 1);
    int cx = I32(a, 0x38) / 300, cy = I32(a, 0x3c) / 300, n = 0;
    uint8_t *tm_sim = g_sim ? PTR(g_sim, 0x68) : NULL, *tm_cl = g_client_obj ? PTR(g_client_obj, 0xf8) : NULL;
    g_nwall_areas++;
    for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
        int ring = (dx < 0 ? -dx : dx) > (dy < 0 ? -dy : dy) ? (dx < 0 ? -dx : dx) : (dy < 0 ? -dy : dy);
        if (ring != 2) continue;
        int corner = (dx == 2 || dx == -2) && (dy == 2 || dy == -2);
        void *td = corner ? fill : wall;
        if (!td) continue;
        int ok = place_one(tm_sim, td, cx + dx, cy + dy);
        if (ok && tm_cl && tm_cl != tm_sim) place_one_force(tm_cl, td, cx + dx, cy + dy);
        if (ok && g_mapview) ((void (*)(void *, void *, int, int))B(0x5459bc))(g_mapview, td, cx + dx, cy + dy);
        n += ok;
    }
    g_nwall_tiles += n; g_last_ring[0] = cx; g_last_ring[1] = cy;
    spawnfix_walldiag("placed");
    if (g_spawn_dbg) LOG("[spawnfix] wall ring at tile %d,%d code %d: %d tiles (sim tm %p, client tm %p, mapview %p)\n", cx, cy, code, n, tm_sim, tm_cl, g_mapview);
}

/* ---- items (BulletExplosionItem +0x80 / SpawnItem +0x110 of the area data): the item factory 0xb55684 is the one the
 * shipped projectile end uses for its own SpawnItem (call at 0xb75ca8). Signature (data, x, y, count, maxItems, dmg, dmg2, v754, v818,
 * v884, owner, source, sim, kind) -> item; afterwards 0xb6e4e4(item, source+0x468, 0, 1) and item+0xd4 = current tick. */
int g_spawn_items = 1;
static long g_nitem;
typedef long (*mkitem_t)(void *, int, int, int, int, int, int, int, int, int, void *, void *, void *, int);
static void make_item(uint8_t *a, uint8_t *pr, void *idata, int x, int y) {
    void *owner = PTR(a, 0x108); if (!owner || !idata) return;
    void *sim = ((void *(*)(void *))B(0xb6a378))(owner);
    int kind = I32(a, 0xf8);
    long it = ((mkitem_t)B(0xb55684))(idata, x, y, 1, 99, I32(a, 0xe8), I32(a, 0xec), I32(a, 0x80), I32(a, 0x84), I32(a, 0x88), owner, pr ? (void *)pr : (void *)a, sim, kind);
    if (!it) return;
    g_nitem++;
    if (pr) {
        I32(it, 0xd4) = ((int (*)(void *))B(0xc27278))(sim);
        ((void (*)(long, void *, int, int))B(0xb6e4e4))(it, pr + 0x468, 0, 1);
    }
}
static int angle_of(int dx, int dy) { return (dx | dy) ? ((i2_t)B(0xd00960))(dx, dy) : 0; }
/* Projectile::end(proj, reason) 0xb75998: reason 1 = range/time end, 2 / 4 = hit a character, 3 = hit a wall, 5 = removed.
 * The shipped code ends with `if (reason != 3 && hasChainBullet) getChainBullet();` (0xb76114..0xb76118: the result is dropped,
 * the server's spawn call is compiled out). Re-implemented for the plain cases (ChainsToEnemies = 0, ExecuteChainSpecialCase 0..2). */
static const int g_ret_parents[] = {
#include "ret_codes.h"
    0};
static int is_return_parent(uint8_t *d) {
    if (!d || !PTR(d, 0x58)) return 0;
    int id = ((int (*)(void *))B(0xac3a80))(PTR(d, 0x58));
    for (int i = 0; g_ret_parents[i]; i++) if (g_ret_parents[i] == id) return 1;
    return 0;
}
static long g_nchain_enemy, g_nchain_ret;
typedef int (*i1p_t)(void *);
/* nearest living enemy hero of projectile `pr` within `range` of (x,y), farther than `minr` (the one just hit) */
static void *nearest_enemy(uint8_t *pr, int x, int y, int range, int minr) {
    uint8_t *mgr = PTR(pr, 0x20); if (!mgr) return NULL;
    void **arr = PTR(mgr, 0x28); int cnt = I32(mgr, 0x34); if (!arr || cnt < 1 || cnt > 512) return NULL;
    int team = I32(pr, 0x84), kind = I32(pr, 0x88); void *best = NULL; long bd = (long)range * range;
    for (int i = 0; i < cnt; i++) {
        uint8_t *o = arr[i]; if (!o || o == pr) continue;
        void **vt = *(void ***)o; if (!vt) continue;
        if (I32(o, 0x88) != kind || I32(o, 0x84) == team) continue;
        if (!(((uint64_t (*)(void *))vt[0x68 / 8])(o) & 1)) continue;
        if (!(((uint64_t (*)(void *))B(0xc70134))(o) & 1)) continue;
        long dx = I32(o, 0x38) - x, dy = I32(o, 0x3c) - y, d2 = dx * dx + dy * dy;
        if (d2 < (long)minr * minr || d2 >= bd) continue;
        bd = d2; best = o;
    }
    return best;
}
static void on_explode(uint64_t *c) {
    uint8_t *pr = (uint8_t *)c[0]; int reason = (int)c[1];
    g_cur_proj = pr; g_cur_tick = g_sim ? I32(g_sim, 0x10) : -1;
    if (g_spawn_dbg && pr && PTR(PTR(pr, 0x10), 0x180)) LOG("[spawnfix] explode call reason %d end90 %d\n", reason, I32(pr, 0x90));
    if (!g_spawn_fix || !g_spawn_chain || !mode() || !pr || rec || I32(pr, 0x90) != 0 || reason == 3 || reason == 5 || reason == 0) return;
    uint8_t *d = PTR(pr, 0x10); if (!d) return;
    uint8_t *child = PTR(d, 0x180); if (!child) return;
    if (g_spawn_dbg) LOG("[spawnfix] explode with chain child: reason %d n %d spec %d toen %d tick %d\n", reason, I32(d, 0x1a4), I32(d, 0x1b4), d[0x1a0] & 1, g_cur_tick);
    int n = I32(d, 0x1a4), spec = I32(d, 0x1b4), spread = I32(d, 0x1ac), cd = I32(d, 0x1b0), toen = d[0x1a0] & 1;
    void *owner = PTR(pr, 0xe0);
    if (!owner) { g_nchain_skip[0]++; return; }
    int x = I32(pr, 0x38), y = I32(pr, 0x3c), sx = I32(pr, 0xfc), sy = I32(pr, 0x100);
    int base = angle_of(x - sx, y - sy), dist = cd * 100;
    void *sim = ((void *(*)(void *))B(0xb6a378))(owner);
    int kind = I32(pr, 0x418), dmg = I32(pr, 0xd4), dmg2 = I32(pr, 0xd8);
    int nspawn = 0, tx[8], ty[8];
    if (toen) {          /* chain to the nearest other enemy within the travel distance */
        void *en = nearest_enemy(pr, x, y, dist > 0 ? dist : 2000, 250);
        if (!en) { g_nchain_skip[1]++; return; }
        tx[0] = I32(en, 0x38); ty[0] = I32(en, 0x3c); nspawn = 1; g_nchain_enemy++;
    } else if (n < 1) {   /* boomerang return to the owner */
        if (!is_return_parent(d)) { g_nchain_skip[2]++; return; }
        tx[0] = I32(owner, 0x38); ty[0] = I32(owner, 0x3c); nspawn = 1; g_nchain_ret++;
    } else {
        if (spec > 3) { g_nchain_skip[3]++; return; }
        nspawn = n > 8 ? 8 : n;
        for (int i = 0; i < nspawn; i++) {
            int ang = nspawn > 1 ? base - spread / 2 + (spread * i) / (nspawn - 1) : base;
            int vx = ((i1_t)B(0xd005e0))(ang + 0x5a), vy = ((i1_t)B(0xd005e0))(ang);
            tx[i] = x + ((vx * dist) >> 10); ty[i] = y + ((vy * dist) >> 10);
        }
    }
    g_nchain++;
    for (int i = 0; i < nspawn; i++) {
        rec = 1;
        void *p = (void *)((spawn_t)B(0xb78190))(x, y, owner, owner, child, tx[i], ty[i], dmg, dmg2, 0, 0, NULL, sim, 0, kind, NULL);
        rec = 0;
        if (p) g_nchain_proj++;
        if (g_spawn_dbg && i < 2) LOG("[spawnfix] chain %d/%d reason %d%s base %d spread %d dist %d -> %d,%d proj %p\n", i, nspawn, reason, toen ? " enemy" : (n < 1 ? " return" : ""), base, spread, dist, tx[i], ty[i], p);
    }
}
static void on_area(uint64_t *c) {
    if (!g_spawn_fix || !mode() || rec) return;
    uint8_t *a = (uint8_t *)c[0]; uint8_t *d = PTR(a, 0x10);
    if (a && d && I32(d, 0x118) == 15 && g_spawn_fix && g_spawn_walls && mode() && !rec) wall_ring(a, d);
    if (a && d && I32(d, 0x118) == 15 && g_spawn_dbg && getenv("NB_SPAWNFIX_TILEDIAG")) {
        uint8_t *cl = (uint8_t *)g_client_obj, *dec = cl ? PTR(cl, 0x38) : NULL, *L = dec ? PTR(dec, 0x10) : NULL;
        void **vt = L ? *(void ***)L : NULL;
        void *ownr = PTR(a, 0x108); void *sm = ownr ? ((void *(*)(void *))B(0xb6a378))(ownr) : NULL;
        LOG("[spawnfix] tile diag: g_sim %p sim(owner) %p client %p; tm(g_sim) %p tm(client) %p tm(sim) %p; dims %d %d %d / %d %d / %d %d\n", g_sim, sm, cl, g_sim ? PTR(g_sim, 0xf8) : NULL, cl ? PTR(cl, 0xf8) : NULL, sm ? PTR(sm, 0xf8) : NULL,
            g_sim && PTR(g_sim, 0xf8) ? I32(PTR(g_sim, 0xf8), 0xc4) : -1, g_sim && PTR(g_sim, 0xf8) ? I32(PTR(g_sim, 0xf8), 0xc8) : -1, g_sim && PTR(g_sim, 0xf8) ? I32(PTR(g_sim, 0xf8), 0xdc) : -1,
            cl && PTR(cl, 0xf8) ? I32(PTR(cl, 0xf8), 0xc4) : -1, cl && PTR(cl, 0xf8) ? I32(PTR(cl, 0xf8), 0xc8) : -1, sm && PTR(sm, 0xf8) ? I32(PTR(sm, 0xf8), 0xc4) : -1, sm && PTR(sm, 0xf8) ? I32(PTR(sm, 0xf8), 0xdc) : -1);
        LOG("[spawnfix] sim+0x68 %p (dims c4 %d c8 %d cc %d d0 %d) client+0xf8 %p\n", g_sim ? PTR(g_sim, 0x68) : NULL, g_sim && PTR(g_sim, 0x68) ? I32(PTR(g_sim, 0x68), 0xc4) : -1, g_sim && PTR(g_sim, 0x68) ? I32(PTR(g_sim, 0x68), 0xc8) : -1, g_sim && PTR(g_sim, 0x68) ? I32(PTR(g_sim, 0x68), 0xcc) : -1, g_sim && PTR(g_sim, 0x68) ? I32(PTR(g_sim, 0x68), 0xd0) : -1, cl ? PTR(cl, 0xf8) : NULL);
        LOG("[spawnfix] WallSpawn area %p data %p client %p dec %p L %p vt %lx slot50 %lx slot38 %lx slot30 %lx pos %d,%d cv %d cv2 %d skinned? a2dc %d\n", a, d, cl, dec, L, vt ? (unsigned long)((uint8_t *)vt - g_libg->base) : 0,
            vt ? (unsigned long)((uint8_t *)vt[10] - g_libg->base) : 0, vt ? (unsigned long)((uint8_t *)vt[7] - g_libg->base) : 0, vt ? (unsigned long)((uint8_t *)vt[6] - g_libg->base) : 0, I32(a, 0x38), I32(a, 0x3c), I32(d, 0x13c), I32(d, 0x140), I32(a, 0x2dc));
    }
    if (!a || !d || I32(d, 0x118) != 5) return;
    uint8_t *bullet = PTR(d, 0x78); void *owner = PTR(a, 0x108);
    int n = I32(d, 0x13c), dist = I32(d, 0x144);
    if (g_spawn_dbg) LOG("[spawnfix] area %p data %p type 5 bullet %p owner %p n %d dist %d off %d pos %d,%d,%d a140 %d a144 %d e8 %d ec %d f8 %d tick %d proj %p\n", a, d, bullet, owner, n, dist, I32(d, 0x148),
        I32(a, 0x38), I32(a, 0x3c), I32(a, 0x40), I32(a, 0x140), I32(a, 0x144), I32(a, 0xe8), I32(a, 0xec), I32(a, 0xf8), g_sim ? I32(g_sim, 0x10) : -1, g_cur_proj);
    g_nbe++;
    if (I32(d, 0x118) == 5 && getenv("NB_SPAWNFIX_TILEDIAG")) {}

    if (g_spawn_items && g_spawn_fix && owner && n >= 1 && n <= 32 && (PTR(d, 0x80) || (!bullet && PTR(d, 0x110)))) {
        uint8_t *pr = (g_cur_proj && g_cur_tick == (g_sim ? I32(g_sim, 0x10) : -1)) ? (uint8_t *)g_cur_proj : NULL;
        void *idata = PTR(d, 0x80);
        int x = I32(a, 0x38), y = I32(a, 0x3c);
        int base0 = pr ? angle_of(I32(pr, 0x38) - I32(pr, 0xfc), I32(pr, 0x3c) - I32(pr, 0x100)) : 0;
        for (int i = 0; i < n; i++) {
            int ang = base0 + (360 * i) / n;
            int vx = ((i1_t)B(0xd005e0))(ang + 0x5a), vy = ((i1_t)B(0xd005e0))(ang);
            make_item(a, pr, idata, x + ((vx * dist * 100) >> 10), y + ((vy * dist * 100) >> 10));
        }
        if (g_spawn_dbg) LOG("[spawnfix] BulletExplosionItem %d items so far %ld\n", n, g_nitem);
    }
    if (!bullet || !owner || n < 1 || n > 32) return;
    void *sim = ((void *(*)(void *))B(0xb6a378))(owner);
    int x = I32(a, 0x38), y = I32(a, 0x3c), kind = I32(a, 0xf8);
    int base = 0, dmg = I32(a, 0xe8);
    if (g_cur_proj && g_cur_tick == (g_sim ? I32(g_sim, 0x10) : -1)) {
        uint8_t *pr = (uint8_t *)g_cur_proj; base = angle_of(I32(pr, 0x38) - I32(pr, 0xfc), I32(pr, 0x3c) - I32(pr, 0x100));
    }
    for (int i = 0; i < n; i++) {
        int ang = base + (360 * i) / n;
        int vx = ((i1_t)B(0xd005e0))(ang + 0x5a), vy = ((i1_t)B(0xd005e0))(ang);
        int tx = x + ((vx * dist * 100) >> 10), ty = y + ((vy * dist * 100) >> 10);
        rec = 1;
        void *p = (void *)((spawn_t)B(0xb78190))(x, y, owner, owner, bullet, tx, ty, dmg, dmg, 0, 0, NULL, sim, 0, kind, NULL);
        rec = 0;
        if (p) g_nbe_proj++;
        if (g_spawn_dbg && i < 2) LOG("[spawnfix]   child %d ang %d -> %d,%d proj %p\n", i, ang, tx, ty, p);
    }
}
void spawnfix_install(void) {
    const char *e = getenv("NB_SPAWNFIX"); if (e) g_spawn_fix = atoi(e);
    if (getenv("NB_SPAWNFIX_DBG")) g_spawn_dbg = 1;
    if ((e = getenv("NB_SPAWNFIX_CHAIN"))) g_spawn_chain = atoi(e);
    if ((e = getenv("NB_SPAWNFIX_WALLS"))) g_spawn_walls = atoi(e);
    if (!g_spawn_fix) { LOG("[spawnfix] disabled\n"); return; }
    hook_enter(g_libg, 0xb75998, on_explode);
    hook_enter(g_libg, 0x53e614, on_mapview);
    hook_enter(g_libg, 0xb29494, on_area);
    LOG("[spawnfix] installed (BulletExplosion children)\n");
}
void spawnfix_status(void) { spawnfix_walldiag("status"); LOG("[spawnfix] BulletExplosion areas %ld, child projectiles %ld; chains %ld (enemy %ld, return %ld, walls %ld areas/%ld tiles, items %ld), child projectiles %ld, skipped (no owner %ld, no enemy %ld, count 0 not a return %ld, special>3 %ld)\n", g_nbe, g_nbe_proj, g_nchain, g_nchain_enemy, g_nchain_ret, g_nwall_areas, g_nwall_tiles, g_nitem, g_nchain_proj, g_nchain_skip[0], g_nchain_skip[1], g_nchain_skip[2], g_nchain_skip[3]); }
