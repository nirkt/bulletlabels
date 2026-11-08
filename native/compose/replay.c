/* replay (rendclone round): re-executes a recorded GL stream (glhost_rec/rec.c) on a headless GLES 3.2 context, with
 * object names remapped (recorded name -> this context's name), and dumps chosen frames.
 *   ./replay rec.nbrec.zst [--frames a,b,c | --every N] [--out DIR] [--max-frame F] [--half]
 * Frames are written as PPM (top-left origin) DIR/f%07d.ppm at the moment of the swap (= what nbgl presented);
 * --half also writes the frame reduced like the collector (2:1 linear blit = 2x2 mean) as DIR/f%07d_h.ppm. */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <zstd.h>
#include "../glhost/nbgl_proto.h"
#define REC_RESULT 0xFFFFFF01u
#define REC_FRAME 0xFFFFFF02u
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline float nb_u2f(uint64_t u) { uint32_t x = (uint32_t)u; float f; memcpy(&f, &x, 4); return f; }
#define IP(i) (A[i] == NBGL_NULL ? NULL : (void *)(cmd + A[i]))
#define OP(i) (A[i] == NBGL_NULL ? NULL : (void *)(reply + A[i]))
static void nb_special(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply, uint64_t *R);
#include "../glhost/nbgl_server_gen.c"
static size_t nb_pix_host(int w, int h, GLenum format, GLenum type);
#include "dump.c"

/* ---- name remapping ---- */
enum { NS_NONE, NS_BUF, NS_TEX, NS_FBO, NS_RBO, NS_VAO, NS_PROG, NS_QUERY, NS_SAMP, NS_XFB, NS_SYNC, NS_N };
#define MAPN (1 << 20)
static uint64_t *g_map[NS_N];
static uint64_t mapn(int ns, uint64_t rec) { if (!rec || rec >= MAPN || !g_map[ns][rec]) return rec; return g_map[ns][rec]; }
static void setmap(int ns, uint64_t rec, uint64_t mine) { if (rec && rec < MAPN) g_map[ns][rec] = mine; }
/* SYNC handles are pointers: a small table */
static uint64_t g_sync_rec[4096], g_sync_mine[4096]; static int g_nsync;
static uint64_t mapsync(uint64_t rec) { for (int i = 0; i < g_nsync; i++) if (g_sync_rec[i] == rec) return g_sync_mine[i]; return 0; }
typedef struct { int16_t op; int8_t arg, ns; } Remap;
static const Remap g_remap[] = {
    {17, 0, NS_PROG}, {17, 1, NS_PROG}, {18, 0, NS_PROG}, {19, 1, NS_BUF}, {20, 1, NS_FBO}, {21, 1, NS_RBO}, {22, 1, NS_TEX},
    {36, 0, NS_PROG}, {46, 0, NS_PROG}, {48, 0, NS_PROG}, {53, 0, NS_PROG}, {53, 1, NS_PROG}, {62, 3, NS_RBO}, {63, 3, NS_TEX},
    {70, 0, NS_PROG}, {71, 0, NS_PROG}, {72, 0, NS_PROG}, {73, 0, NS_PROG}, {80, 0, NS_PROG}, {81, 0, NS_PROG}, {83, 0, NS_PROG},
    {84, 0, NS_PROG}, {86, 0, NS_PROG}, {90, 0, NS_PROG}, {91, 0, NS_PROG}, {92, 0, NS_PROG}, {97, 0, NS_BUF}, {99, 0, NS_FBO},
    {100, 0, NS_PROG}, {101, 0, NS_RBO}, {102, 0, NS_PROG}, {103, 0, NS_TEX}, {105, 0, NS_PROG}, {114, 0, NS_PROG}, {146, 0, NS_PROG},
    {147, 0, NS_PROG}, {167, 0, NS_QUERY}, {168, 1, NS_QUERY}, {171, 0, NS_QUERY}, {183, 2, NS_TEX}, {186, 0, NS_VAO}, {189, 0, NS_VAO},
    {193, 2, NS_BUF}, {194, 2, NS_BUF}, {195, 0, NS_PROG}, {196, 0, NS_PROG}, {204, 0, NS_PROG}, {205, 0, NS_PROG}, {220, 0, NS_PROG},
    {221, 0, NS_PROG}, {222, 0, NS_PROG}, {223, 0, NS_PROG}, {224, 0, NS_PROG}, {225, 0, NS_PROG}, {229, 0, NS_SYNC}, {230, 0, NS_SYNC},
    {231, 0, NS_SYNC}, {232, 0, NS_SYNC}, {234, 0, NS_SYNC}, {239, 0, NS_SAMP}, {240, 1, NS_SAMP}, {241, 0, NS_SAMP}, {242, 0, NS_SAMP},
    {243, 0, NS_SAMP}, {244, 0, NS_SAMP}, {245, 0, NS_SAMP}, {246, 0, NS_SAMP}, {248, 1, NS_XFB}, {251, 0, NS_XFB}, {254, 0, NS_PROG},
    {255, 0, NS_PROG}, {256, 0, NS_PROG}, {0, 0, 0}};
static int g_gen_ns(unsigned op) {
    switch (op) { case 65: return NS_BUF; case 67: return NS_FBO; case 68: return NS_RBO; case 69: return NS_TEX; case 165: return NS_QUERY;
    case 188: return NS_VAO; case 237: return NS_SAMP; case 250: return NS_XFB; case 41: case 42: return NS_PROG; case 228: return NS_SYNC; }
    return 0;
}
static int g_del_ns(unsigned op) {
    switch (op) { case 44: return NS_BUF; case 45: return NS_FBO; case 47: return NS_RBO; case 49: return NS_TEX; case 166: return NS_QUERY;
    case 187: return NS_VAO; case 238: return NS_SAMP; case 249: return NS_XFB; }
    return 0;
}
static const Remap *g_rtab[300][4];
static void remap_init(void) {
    for (int i = 0; i < NS_N; i++) g_map[i] = calloc(MAPN, 8);
    for (const Remap *r = g_remap; r->op; r++) { int k = 0; while (g_rtab[r->op][k]) k++; g_rtab[r->op][k] = r; }
}
static void remap_args(unsigned op, uint64_t *A, uint8_t *cmd) {
    if (op < 300) for (int k = 0; k < 4 && g_rtab[op][k]; k++) {
        const Remap *r = g_rtab[op][k];
        A[r->arg] = r->ns == NS_SYNC ? mapsync(A[r->arg]) : mapn(r->ns, A[r->arg]);
    }
    int dn = g_del_ns(op);
    if (dn && A[1] != NBGL_NULL) { uint32_t *v = (uint32_t *)(cmd + A[1]); for (uint64_t i = 0; i < A[0]; i++) v[i] = (uint32_t)mapn(dn, v[i]); }
}

static uint8_t g_reply[NBGL_REPLY]; static long g_steps_frame = -1; static int g_noppm; static int g_stepn;
static unsigned g_last_op; static uint64_t g_last_R; static uint32_t g_last_names[4096]; static int g_last_n;
static void nb_special(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply, uint64_t *R) {
    const char *n = nb_op_name(op); (void)R;
    if (!n) return;
    if (!strcmp(n, "glPixelStorei")) glPixelStorei((GLenum)A[0], (GLint)A[1]);
    else if (!strcmp(n, "glShaderSource")) { const char *s = IP(2); glShaderSource((GLuint)A[0], 1, &s, NULL); }
    else if (!strcmp(n, "glTransformFeedbackVaryings") || !strcmp(n, "glGetUniformIndices")) {
        const char *v[256]; const char *p = IP(2); int cnt = (int)A[1];
        for (int i = 0; i < cnt && i < 256; i++) { v[i] = p; p += strlen(p) + 1; }
        if (n[2] == 'T') glTransformFeedbackVaryings((GLuint)A[0], cnt, v, (GLenum)A[3]);
        else glGetUniformIndices((GLuint)A[0], cnt, v, OP(3));
    }
}

/* ---- zstd record reader ---- */
static FILE *g_f; static ZSTD_DCtx *g_z; static uint8_t *g_in; static size_t g_inn, g_inpos, g_incap;
static uint8_t *g_buf; static size_t g_bufn, g_bufpos, g_bufcap;
static int fill(size_t need) {
    while (g_bufn - g_bufpos < need) {
        if (g_bufpos) { memmove(g_buf, g_buf + g_bufpos, g_bufn - g_bufpos); g_bufn -= g_bufpos; g_bufpos = 0; }
        if (g_bufcap - g_bufn < (8u << 20)) { g_bufcap = g_bufcap * 2 + need; g_buf = realloc(g_buf, g_bufcap); }
        if (g_inpos == g_inn) { g_inn = fread(g_in, 1, g_incap, g_f); g_inpos = 0; if (!g_inn) return 0; }
        ZSTD_inBuffer in = {g_in, g_inn, g_inpos}; ZSTD_outBuffer out = {g_buf + g_bufn, g_bufcap - g_bufn, 0};
        size_t r = ZSTD_decompressStream(g_z, &out, &in); if (ZSTD_isError(r)) { fprintf(stderr, "zstd: %s\n", ZSTD_getErrorName(r)); return 0; }
        g_inpos = in.pos; g_bufn += out.pos;
    }
    return 1;
}

static void write_ppm(const char *path, const uint8_t *rgba, int w, int h) {
    FILE *f = fopen(path, "wb"); fprintf(f, "P6\n%d %d\n255\n", w, h);
    uint8_t *row = malloc(w * 3);
    for (int y = h - 1; y >= 0; y--) { for (int x = 0; x < w; x++) memcpy(row + 3 * x, rgba + ((size_t)y * w + x) * 4, 3); fwrite(row, 1, w * 3, f); }
    free(row); fclose(f);
}

/* ---- composite mode (bulletlabels): during frame g_comp, the default framebuffer first receives a background image
 * (another capture's finished frame) and only the listed on-screen draws (1-based, in order, as the ID pass numbers them)
 * are executed: one attack's draws blended onto another frame by the game's own shaders and blend state. ---- */
static long g_comp = -1; static int g_comp_n = 0, g_comp_bg_done = 0; static uint8_t *g_keep = NULL; static int g_keep_all = 0;
static uint8_t *g_bg = NULL; static int g_bgw = 0, g_bgh = 0;
static uint8_t *read_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return NULL; }
    int mx; if (fscanf(f, "P6 %d %d %d", w, h, &mx) != 3) { fclose(f); return NULL; } fgetc(f);
    uint8_t *rgb = malloc((size_t)*w * *h * 3), *rgba = malloc((size_t)*w * *h * 4);
    if (fread(rgb, 1, (size_t)*w * *h * 3, f) != (size_t)*w * *h * 3) { fclose(f); return NULL; } fclose(f);
    for (int y = 0; y < *h; y++) for (int x = 0; x < *w; x++) {   /* to GL rows (bottom-up) */
        const uint8_t *s = rgb + ((size_t)y * *w + x) * 3; uint8_t *d = rgba + ((size_t)(*h - 1 - y) * *w + x) * 4;
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255; }
    free(rgb); return rgba;
}
static void comp_put_bg(void) {   /* the background into the default framebuffer, GL state restored */
    GLint pd = 0, pr = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd); glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr);
    GLint at = 0, tb = 0; glGetIntegerv(GL_ACTIVE_TEXTURE, &at); glGetIntegerv(GL_TEXTURE_BINDING_2D, &tb);
    GLint ub = 0; glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &ub); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    GLint ua = 4, rl = 0; glGetIntegerv(GL_UNPACK_ALIGNMENT, &ua); glGetIntegerv(GL_UNPACK_ROW_LENGTH, &rl); glPixelStorei(GL_UNPACK_ALIGNMENT, 1); glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    GLboolean sc = glIsEnabled(GL_SCISSOR_TEST); if (sc) glDisable(GL_SCISSOR_TEST);
    GLuint t, fb; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, g_bgw, g_bgh, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_bg);
    glGenFramebuffers(1, &fb); glBindFramebuffer(GL_READ_FRAMEBUFFER, fb); glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, g_bgw, g_bgh, 0, 0, g_bgw, g_bgh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd); glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, tb); glActiveTexture(at); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ub);
    glPixelStorei(GL_UNPACK_ALIGNMENT, ua); glPixelStorei(GL_UNPACK_ROW_LENGTH, rl); if (sc) glEnable(GL_SCISSOR_TEST);
}

int main(int argc, char **argv) {
    const char *path = NULL, *outdir = "."; long every = 0, maxf = -1; int half = 0; long want[4096]; int nwant = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) { for (char *t = strtok(argv[++i], ","); t && nwant < 4096; t = strtok(NULL, ",")) want[nwant++] = atol(t); }
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atol(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--max-frame") && i + 1 < argc) maxf = atol(argv[++i]);
        else if (!strcmp(argv[i], "--half")) half = 1;
        else if (!strcmp(argv[i], "--no-ppm")) g_noppm = 1;
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) g_steps_frame = atol(argv[++i]);
        else if (!strcmp(argv[i], "--draws") && i + 1 < argc) g_dumpdir = argv[++i];
        else if (!strcmp(argv[i], "--comp") && i + 1 < argc) g_comp = atol(argv[++i]);
        else if (!strcmp(argv[i], "--bg") && i + 1 < argc) { g_bg = read_ppm(argv[++i], &g_bgw, &g_bgh); if (!g_bg) return 1; }
        else if (!strcmp(argv[i], "--keep") && i + 1 < argc) { g_keep = calloc(65536, 1); const char *k = argv[++i];
            if (!strcmp(k, "all")) g_keep_all = 1; else if (strcmp(k, "none")) for (char *t = strtok((char *)k, ","); t; t = strtok(NULL, ",")) { long v = atol(t); if (v > 0 && v < 65536) g_keep[v] = 1; } }
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: replay rec.zst [--frames a,b] [--every N] [--out DIR] [--max-frame F] [--half]\n"); return 1; }
    g_f = fopen(path, "rb"); if (!g_f) { perror(path); return 1; }
    g_z = ZSTD_createDCtx(); g_incap = ZSTD_DStreamInSize() * 16; g_in = malloc(g_incap); g_bufcap = 256u << 20; g_buf = malloc(g_bufcap);
    if (!fill(16) || memcmp(g_buf, "NBREC001", 8)) { fprintf(stderr, "bad header\n"); return 1; }
    int W, H; memcpy(&W, g_buf + 8, 4); memcpy(&H, g_buf + 12, 4); g_bufpos = 16;

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (gpd) { EGLDisplay d2 = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL); if (d2) dpy = d2; }
    EGLint ma, mi; if (!eglInitialize(dpy, &ma, &mi)) { fprintf(stderr, "eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ca[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
                   EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg; EGLint n; if (!eglChooseConfig(dpy, ca, &cfg, 1, &n) || n < 1) { fprintf(stderr, "no config\n"); return 1; }
    EGLint pa[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_NONE}; EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pa);
    EGLint xa[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    if (!eglMakeCurrent(dpy, surf, surf, ctx)) { fprintf(stderr, "eglMakeCurrent failed\n"); return 1; }
    { EGLint p1[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE}; g_xdpy = dpy; g_xmain = ctx; g_xmsurf = surf; g_xsurf = eglCreatePbufferSurface(dpy, cfg, p1);
      g_xctx = eglCreateContext(dpy, cfg, ctx, xa); if (!g_xctx) fprintf(stderr, "no shared context\n"); }
    nb_load_procs(); remap_init();
    if (g_dumpdir) { char p[1024]; mkdir(g_dumpdir, 0755); snprintf(p, sizeof p, "%s/blobs", g_dumpdir); mkdir(p, 0755); snprintf(p, sizeof p, "%s/shaders", g_dumpdir); mkdir(p, 0755); snprintf(p, sizeof p, "%s/tex", g_dumpdir); mkdir(p, 0755); }
    fprintf(stderr, "[replay] %s, %dx%d, %s\n", path, W, H, glGetString(GL_RENDERER));

    uint8_t *px = malloc((size_t)W * H * 4), *hp = malloc((size_t)W * H);
    long frame = -1, ncmd = 0, ndraw = 0, nmiss = 0; double t0 = now(), tf = t0;
    uint8_t *cmdcopy = malloc(64u << 20); size_t cmdcap = 64u << 20;
    while (fill(8)) {
        uint32_t opf = *(uint32_t *)(g_buf + g_bufpos), len = *(uint32_t *)(g_buf + g_bufpos + 4);
        if (len < 8 || !fill(len)) break;
        uint8_t *rec = g_buf + g_bufpos; g_bufpos += len;
        if (opf == REC_FRAME) { memcpy(&frame, rec + 8, 8); g_frame_now = frame + 1; continue; }
        if (opf == REC_RESULT) {
            uint32_t op = *(uint32_t *)(rec + 8), cnt = *(uint32_t *)(rec + 12); uint64_t R; memcpy(&R, rec + 16, 8);
            int ns = g_gen_ns(op);
            if (op == 228) { if (g_nsync < 4096) { g_sync_rec[g_nsync] = R; g_sync_mine[g_nsync] = g_last_R; g_nsync++; } }
            else if (op == 41 || op == 42) setmap(NS_PROG, R, g_last_R);
            else { const uint32_t *nm = (const uint32_t *)(rec + 24); for (uint32_t i = 0; i < cnt && (int)i < g_last_n; i++) { setmap(ns, nm[i], g_last_names[i]); if (nm[i] != g_last_names[i]) nmiss++; } }
            continue;
        }
        unsigned op = opf & ~NBGL_SYNC;
        if (op == NBGL_OP_WRAP || op == NBGL_OP_MAPREAD || op == NBGL_OP_CAPTURE) continue;
        if (op == NBGL_OP_SWAP) {
            int dump = 0;
            if (every && frame % every == 0) dump = 1;
            for (int i = 0; i < nwant; i++) if (want[i] == frame) dump = 1;
            if (dump && !g_noppm) {
                GLint pd = 0, pr = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd); glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr);
                GLboolean sc = glIsEnabled(GL_SCISSOR_TEST); if (sc) glDisable(GL_SCISSOR_TEST);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
                char p[1024]; snprintf(p, sizeof p, "%s/f%07ld.ppm", outdir, frame); write_ppm(p, px, W, H);
                if (half) {   /* 2x2 mean with GL_LINEAR rounding: (a+b+c+d)/4 in float, rounded */
                    int w2 = W / 2, h2 = H / 2; uint8_t *q = malloc((size_t)w2 * h2 * 4);
                    for (int y = 0; y < h2; y++) for (int x = 0; x < w2; x++) for (int c = 0; c < 4; c++) {
                        const uint8_t *s = px + ((size_t)(2 * y) * W + 2 * x) * 4 + c;
                        q[((size_t)y * w2 + x) * 4 + c] = (uint8_t)((s[0] + s[4] + s[(size_t)W * 4] + s[(size_t)W * 4 + 4] + 2) / 4);
                    }
                    snprintf(p, sizeof p, "%s/f%07ld_h.ppm", outdir, frame); write_ppm(p, q, w2, h2); free(q);
                }
                if (sc) glEnable(GL_SCISSOR_TEST);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd);
            }
            if (g_dj) dump_end_frame();
            eglSwapBuffers(dpy, surf);
            if (g_dumpdir) { int dn = every && (frame + 1) % every == 0; for (int i = 0; i < nwant; i++) if (want[i] == frame + 1) dn = 1; if (dn) dump_begin_frame(frame + 1); }
            if (frame % 200 == 0) { double t = now(); fprintf(stderr, "[replay] frame %ld cmds %ld draws %ld (%.1f fps) name-mismatches %ld\n", frame, ncmd, ndraw, 200 / (t - tf), nmiss); tf = t; }
            if (maxf >= 0 && frame >= maxf) break;
            continue;
        }
        if (len > cmdcap) { cmdcap = len; cmdcopy = realloc(cmdcopy, cmdcap); }
        memcpy(cmdcopy, rec, len);
        uint64_t *A = (uint64_t *)(cmdcopy + 8);
        remap_args(op, A, cmdcopy);
        uint64_t R = 0;
        if (g_dumpdir) { shadow_pre(op, A, cmdcopy); texver_pre(op, A); eth_pre(op, A, cmdcopy, len);
            if (op == 56 || op == 57 || op == 159 || op == 226 || op == 227 || op == 31 || op == 181 || (op >= 214 && op <= 217)) texver_draw(frame + 1); if (op == 105 || op == 255) { if (A[0] < SH_MAXB) g_pi[A[0]].valid = 0; } }
        if (g_dj && (op == 56 || op == 57 || op == 159 || op == 226 || op == 227)) dump_draw(op, A, cmdcopy);
        if (g_dj && (op == 31 || (op >= 214 && op <= 217))) dump_clear(op, A, cmdcopy);
        if (op == 255) {   /* glProgramBinary: the recorder made it fail; do the same */
            GLsizei bl = (GLsizei)A[3]; uint8_t *junk = calloc(1, bl > 16 ? bl : 16); glProgramBinary((GLuint)A[0], (GLenum)A[1], junk, bl > 16 ? bl : 16); free(junk);
        } else if (op >= 16 && op != 87 && op != 218) {
            int skip = 0;
            if (g_comp >= 0 && frame + 1 == g_comp && (op == 56 || op == 57 || op == 159 || op == 226 || op == 227)) {
                GLint pd = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd);
                if (pd == 0) {
                    if (!g_comp_bg_done && g_bg) { comp_put_bg(); g_comp_bg_done = 1; }
                    g_comp_n++; if (!g_keep_all && !(g_keep && g_comp_n < 65536 && g_keep[g_comp_n])) skip = 1;
                }
            }
            if (!skip) R = nb_dispatch(op, A, cmdcopy, g_reply);
        }
        if (op == 56 || op == 57 || op == 159 || op == 226 || op == 227) {
            ndraw++;
            if (frame + 1 == g_steps_frame) {   /* after each draw of this frame: the bound draw framebuffer (viewport region of its attachment) */
                GLint pd = 0, pr = 0, vp[4]; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd); glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr);
                GLboolean sc = glIsEnabled(GL_SCISSOR_TEST); if (sc) glDisable(GL_SCISSOR_TEST);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, pd); glPixelStorei(GL_PACK_ALIGNMENT, 1);
                int rw = W, rh = H; if (pd) { rw = 1024; rh = 1024; GLint ow = 0, oh = 0, ty = 0, nm = 0;
                    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &ty);
                    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &nm);
                    if (ty == GL_TEXTURE && nm < SH_MAXB && g_st[nm].w) { ow = g_st[nm].w; oh = g_st[nm].ht; } if (ow) { rw = ow; rh = oh; } }
                GLint prb = 0; if (pd) { glGetIntegerv(GL_READ_BUFFER, &prb); glReadBuffer(GL_COLOR_ATTACHMENT0); }
                (void)vp; glReadPixels(0, 0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, px);
                if (pd) glReadBuffer(prb);
                char p[1024]; snprintf(p, sizeof p, "%s/s%07ld_%03d_fb%d.ppm", outdir, frame + 1, g_stepn++, pd);
                const char *cr = getenv("NB_STEPS_CROP"); int cx, cy, cw, ch;   /* x,y,w,h in GL (bottom-up) px of the main fb only */
                if (cr && sscanf(cr, "%d,%d,%d,%d", &cx, &cy, &cw, &ch) == 4) {
                    if (!pd) { for (int r = 0; r < ch; r++) memmove(px + (size_t)r * cw * 4, px + ((size_t)(cy + r) * rw + cx) * 4, (size_t)cw * 4); write_ppm(p, px, cw, ch); }
                } else write_ppm(p, px, rw, rh);
                if (sc) glEnable(GL_SCISSOR_TEST); glBindFramebuffer(GL_READ_FRAMEBUFFER, pr);
            }
        }
        int gns = g_gen_ns(op);
        if (gns) { g_last_R = R; g_last_n = 0;
            if (op != 41 && op != 42 && op != 228 && A[1] != NBGL_NULL) { g_last_n = (int)A[0] < 4096 ? (int)A[0] : 4096; memcpy(g_last_names, g_reply + A[1], 4 * g_last_n); } }
        ncmd++;
    }
    if (g_dumpdir) fprintf(stderr, "[replay] buffer reads differing from the shadow: %ld\n", g_shadow_bad);
    if (g_bdat) { fclose(g_bdat); fclose(g_bidx); }
    fprintf(stderr, "[replay] done: frame %ld, %ld cmds, %ld draws, %.1f s, name mismatches %ld\n", frame, ncmd, ndraw, now() - t0, nmiss);
    return 0;
}
