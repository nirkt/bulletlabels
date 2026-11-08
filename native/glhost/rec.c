/* rec.c (rendclone round): records the game's GL command stream as nbgl receives it, for offline replay / analysis.
 *   NBGL_REC=<path>    write <path> (zstd stream) with every ring command in order
 *   NBGL_REC_LEVEL=N   zstd level (default 2)
 * Record layout (after the 16-byte header "NBREC001", u32 W, u32 H): the raw ring commands (u32 op|SYNC, u32 len, args,
 * blobs; WRAP markers dropped) plus extra records:
 *   REC_RESULT (op 0xFFFFFF01): u32 op, u32 len, u32 orig_op, u32 n, u64 R, u32 names[n]  (object names / handles that
 *       glGen*, glCreate*, glFenceSync returned: the replayer maps recorded names to its own)
 *   REC_FRAME  (op 0xFFFFFF02): u32 op, u32 len, u64 frame (swap count before this swap), f64 t (s since start)
 * glProgramBinary is always made to fail (junk binary) so the game compiles every program from source. Included by nbgl.c. */
#include <zstd.h>
#define REC_RESULT 0xFFFFFF01u
#define REC_FRAME 0xFFFFFF02u
static ZSTD_CCtx *g_rec_z; static uint8_t *g_rec_out; static size_t g_rec_outn; static double g_rec_t0;
static uint64_t g_rec_bytes, g_rec_zbytes;
static uint8_t *g_rec_buf; static size_t g_rec_bufn, g_rec_bufcap;
static void rec_flush(int end) {
    ZSTD_inBuffer in = {g_rec_buf, g_rec_bufn, 0};
    for (;;) {
        ZSTD_outBuffer out = {g_rec_out, g_rec_outn, 0};
        size_t r = ZSTD_compressStream2(g_rec_z, &out, &in, end ? ZSTD_e_end : ZSTD_e_continue);
        if (ZSTD_isError(r)) { fprintf(stderr, "[rec] zstd error %s\n", ZSTD_getErrorName(r)); break; }
        if (out.pos) { fwrite(g_rec_out, 1, out.pos, g_rec_f); g_rec_zbytes += out.pos; }
        if (end ? r == 0 : in.pos == in.size) break;
    }
    g_rec_bufn = 0;
}
static void rec_put(const void *p, size_t n) {
    if (!g_rec_f) return;
    if (g_rec_bufn + n > g_rec_bufcap) { rec_flush(0); if (n > g_rec_bufcap) { g_rec_bufcap = n; g_rec_buf = realloc(g_rec_buf, n); } }
    memcpy(g_rec_buf + g_rec_bufn, p, n); g_rec_bufn += n; g_rec_bytes += n;
}
static void rec_init(int W, int H) {
    const char *p0 = getenv("NBGL_REC"); if (!p0 || !*p0) return;
    char pb[1024]; const char *p = p0;   /* "%s" in the path = the shm basename (one file per renderer) */
    if (strstr(p0, "%s")) { const char *b = g_shm_path ? strrchr(g_shm_path, '/') : NULL; b = b ? b + 1 : "nbgl"; snprintf(pb, sizeof pb, p0, b); p = pb; }
    g_rec_f = fopen(p, "wb"); if (!g_rec_f) { perror(p); return; }
    g_rec_z = ZSTD_createCCtx(); const char *l = getenv("NBGL_REC_LEVEL");
    ZSTD_CCtx_setParameter(g_rec_z, ZSTD_c_compressionLevel, l ? atoi(l) : 2);
    ZSTD_CCtx_setParameter(g_rec_z, ZSTD_c_nbWorkers, 2);
    g_rec_outn = ZSTD_CStreamOutSize() * 8; g_rec_out = malloc(g_rec_outn);
    g_rec_bufcap = 64u << 20; g_rec_buf = malloc(g_rec_bufcap);
    uint8_t h[16] = "NBREC001"; memcpy(h + 8, &W, 4); memcpy(h + 12, &H, 4); rec_put(h, 16);
    g_rec_t0 = now(); fprintf(stderr, "[rec] recording the GL stream to %s\n", p);
}
static void rec_cmd(const uint8_t *cmd, uint32_t len) { if (g_rec_f) rec_put(cmd, len); }
/* after dispatch: names / handles the call returned */
static void rec_result(unsigned op, const uint64_t *A, const uint8_t *reply, uint64_t R) {
    if (!g_rec_f) return;
    uint32_t n = 0; const uint32_t *names = NULL;
    switch (op) {
    case 65: case 67: case 68: case 69: case 165: case 188: case 237: case 250:   /* glGen* (n, out names) */
        n = (uint32_t)A[0]; names = A[1] == NBGL_NULL ? NULL : (const uint32_t *)(reply + A[1]); if (!names) n = 0; break;
    case 41: case 42: case 228: break;                                             /* glCreateProgram / Shader, glFenceSync: R */
    default: return;
    }
    uint32_t hdr[4] = {REC_RESULT, (uint32_t)(24 + ((4 * n + 7) & ~7u)), op, n};
    rec_put(hdr, 16); rec_put(&R, 8);
    if (n) { rec_put(names, 4 * n); if (n & 1) { uint32_t z = 0; rec_put(&z, 4); } }
}
static void rec_frame(uint64_t frame) {
    if (!g_rec_f) return;
    double t = now() - g_rec_t0; uint32_t hdr[2] = {REC_FRAME, 24};
    rec_put(hdr, 8); rec_put(&frame, 8); rec_put(&t, 8);
    if ((frame % 300) == 0) {
        rec_flush(0); fflush(g_rec_f);
        fprintf(stderr, "[rec] frame %lu: %.1f MB raw, %.1f MB compressed\n", (unsigned long)frame, g_rec_bytes / 1e6, g_rec_zbytes / 1e6);
    }
}
static void rec_close(void) {
    if (!g_rec_f) return;
    rec_flush(1); fclose(g_rec_f); g_rec_f = NULL;
    fprintf(stderr, "[rec] closed: %.1f MB raw, %.1f MB compressed\n", g_rec_bytes / 1e6, g_rec_zbytes / 1e6);
}
