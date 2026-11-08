/* idpass (expertbot round, renderer labels): an ID buffer for every frame drawn into the default framebuffer.
 * IN-PLACE hook (v2; the twin-program version, idpass_twin.c, copied state badly for some models): right after the game
 * links a program, its fragment shader is replaced by a rewritten one and the GAME'S OWN program object is relinked:
 *   main() renamed to nb_om(); new main(): nb_om(); if (nb_idc.a > 0.5) { if (OUT.a < nb_athr) discard; OUT = nb_idc; }
 * All-zero uniforms (GL's default, and what the game may reset them to) = pass-through: the game renders as before. For a dumped
 * frame, after each draw into framebuffer 0 the same command is re-dispatched into a private ID framebuffer (same size,
 * own depth/stencil) with nb_idc = the draw's ID colour, blending off and the alpha test on for blended draws; then the
 * defaults are restored. Every uniform, texture and buffer binding is the game's own. Depth-only programs (no colour
 * output) are replayed into the ID buffer with colour writes off, so later depth-tested passes see the same depth.
 * Clears of framebuffer 0 clear the ID buffer too. Every NBGL_ID_EVERY-th frame (NBGL_ID_DIR set): the frame (RGBA),
 * the ID buffer (id = r + 256 g + 65536 b, 0 = nothing), the frame's draws (JSON lines), the camera matrix.
 * NBGL_ID_COLOUR=1 (debug): the ID buffer gets the game's own colours (should equal the frame). Included by nbgl.c. */
#include <sys/stat.h>
#include <zlib.h>
#define ID_MAXP 8192
typedef struct { int hooked, depth_only, es3; GLint loc_idc, loc_athr, loc_ivp, loc_vpa, loc_cm; uint32_t vs_h, fs_h; char *vsrc, *fsrc; } IdProg;
static IdProg g_idp[ID_MAXP];
static int g_id_on = -1, g_id_every = 20, g_id_active, g_id_ndraw, g_id_frame_dumps, g_id_colour; static const char *g_id_dir;
static GLuint g_id_fbo, g_id_col, g_id_dep, g_wp_fbo, g_wp_col, g_wp_dep; static int g_id_w, g_id_h; static float g_ivp[16]; static int g_wp_on = 1; static GLuint g_id_cur_prog, g_id_cur_dfb;
static unsigned OP_PBIN, OP_LINK, OP_USE, OP_BINDFB, OP_CLEAR, OP_DA, OP_DE, OP_DAI, OP_DEI, OP_DRE, OP_BSD;
/* texture identity across runs: content hash of each 2D texture's level-0 upload (or its first sub-upload after glTexStorage2D /
 * a NULL upload); logged per draw ("th") so draws can be classified by WHAT they draw (UI bars, font, sprite atlases) */
static unsigned OP_TI, OP_CTI, OP_TS, OP_TSI, OP_CTSI; static uint32_t g_id_cmdlen;
#define ID_MAXT 65536
typedef struct { uint64_t h; int w, ht; } IdTex; static IdTex g_idt[ID_MAXT];
/* on-demand mode (NBGL_ID_EVERY=0): every frame is rendered into a ring slot (ID + eye-depth framebuffers, the draw list
 * in memory); the collector asks for frames by their swap number through <shm>.idreq (IdReq), and requested frames still
 * in the ring are written out at the next frame end (or when idle) */
#define ID_RING 4
typedef struct { GLuint id_fbo, id_col, id_dep, uv_col, wp_fbo, wp_col, wp_dep, rgb_fbo, rgb_col; long frame; char *js; size_t jsn; FILE *jf; float cam[16], anchor[2], wref; int ndraw, valid, w, h; } IdSlot;
typedef struct { volatile int64_t w, r; int64_t f[1000]; } IdReq;
static int g_id_keep_add;      /* NBGL_ID_ADDITIVE=1 keeps additive draws (dst ONE: glows, halos, trails, light) in the ID pass; default: skipped = transparent */
static float g_athr = 0.1f;   /* renderer-side cut on coverage; per-class cuts offline (coverage channel) */
static IdSlot g_ring[ID_RING]; static int g_ondemand; static IdReq *g_req; static long g_req_served, g_req_late;
static FILE *g_id_json; static float g_id_cam[16]; static int g_id_have_cam; static long g_id_hooked, g_id_failed;

static uint32_t fnv(const char *s) { uint32_t h = 2166136261u; for (; s && *s; s++) h = (h ^ (uint8_t)*s) * 16777619u; return h; }
static void id_init(void) {
    const char *e = getenv("NBGL_ID_DIR"); g_id_on = e && *e;
    if (g_id_on) {   /* one subdirectory per renderer: <dir>/<basename of the shm path> */
        mkdir(e, 0755); const char *b = g_shm_path ? strrchr(g_shm_path, '/') : NULL; b = b ? b + 1 : (g_shm_path ? g_shm_path : "nbgl");
        static char dir[600]; snprintf(dir, sizeof dir, "%s/%s", e, b); g_id_dir = dir;
    }
    if ((e = getenv("NBGL_ID_EVERY"))) { g_id_every = atoi(e) > 0 ? atoi(e) : 20; g_ondemand = atoi(e) == 0; }
    if (g_id_on && g_ondemand && g_shm_path) {
        char rp[600]; snprintf(rp, sizeof rp, "%s.idreq", g_shm_path);
        int fd = open(rp, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0 && !ftruncate(fd, sizeof(IdReq))) { g_req = mmap(NULL, sizeof(IdReq), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (g_req == MAP_FAILED) g_req = NULL; }
        if (fd >= 0) close(fd);
        fprintf(stderr, "[idpass] on-demand mode: requests via %s (%s)\n", rp, g_req ? "ok" : "FAILED");
    }
    if ((e = getenv("NBGL_ID_COLOUR"))) g_id_colour = atoi(e);
    if ((e = getenv("NBGL_ID_ATHR"))) g_athr = (float)atof(e);
    if ((e = getenv("NBGL_ID_ADDITIVE"))) g_id_keep_add = atoi(e);
    for (unsigned op = 0; op < 2048; op++) {
        const char *n = nb_op_name(op); if (!n) continue;
        if (!strcmp(n, "glProgramBinary")) OP_PBIN = op;
        else if (!strcmp(n, "glLinkProgram")) OP_LINK = op; else if (!strcmp(n, "glUseProgram")) OP_USE = op;
        else if (!strcmp(n, "glBindFramebuffer")) OP_BINDFB = op; else if (!strcmp(n, "glClear")) OP_CLEAR = op;
        else if (!strcmp(n, "glDrawArrays")) OP_DA = op; else if (!strcmp(n, "glDrawElements")) OP_DE = op;
        else if (!strcmp(n, "glDrawArraysInstanced")) OP_DAI = op; else if (!strcmp(n, "glDrawElementsInstanced")) OP_DEI = op;
        else if (!strcmp(n, "glDrawRangeElements")) OP_DRE = op; else if (!strcmp(n, "glBufferSubData")) OP_BSD = op;
        else if (!strcmp(n, "glTexImage2D")) OP_TI = op; else if (!strcmp(n, "glCompressedTexImage2D")) OP_CTI = op;
        else if (!strcmp(n, "glTexStorage2D")) OP_TS = op; else if (!strcmp(n, "glTexSubImage2D")) OP_TSI = op;
        else if (!strcmp(n, "glCompressedTexSubImage2D")) OP_CTSI = op;
    }
    if (g_id_on) { mkdir(g_id_dir, 0755); fprintf(stderr, "[idpass] on (in-place): dir %s every %d frames\n", g_id_dir, g_id_every); }
}
/* the fragment shader with the ID switch; NULL if it has no colour output (depth-only) or cannot be parsed */
static char *id_frag_src(const char *src) {
    const char *decl_ls = NULL;
    const char *m = strstr(src, "void main()"); if (!m) m = strstr(src, "void main(");
    if (!m) return NULL;
    char outname[64] = "gl_FragColor"; int es3 = strstr(src, "#version 300") || strstr(src, "#version 310") || strstr(src, "#version 320");
    if (es3) {   /* the first global "out ... vec4 NAME;" before main (not a function parameter) */
        const char *p = src; int found = 0; decl_ls = NULL;
        while ((p = strstr(p, "out ")) && p < m) {
            const char *ls = p; while (ls > src && ls[-1] != '\n') ls--;
            int depth = 0; for (const char *q = src; q < ls; q++) { if (*q == '{') depth++; else if (*q == '}') depth--; }
            const char *v = strstr(p, "vec4"), *semi = strchr(p, ';'), *paren = strchr(ls, '(');
            int in_params = paren && paren < p && strchr(ls, ')') && strchr(ls, ')') > p && strncmp(ls, "layout", 6);
            if (depth == 0 && !in_params && v && semi && v < semi) {
                const char *q = v + 4; while (*q == ' ') q++; int k = 0;
                while ((q[k] == '_' || (q[k] >= 'a' && q[k] <= 'z') || (q[k] >= 'A' && q[k] <= 'Z') || (q[k] >= '0' && q[k] <= '9')) && k < 63) { outname[k] = q[k]; k++; }
                outname[k] = 0; found = 1; decl_ls = ls; break;
            }
            p += 4;
        }
        if (!found) return NULL;
    } else if (!strstr(src, "gl_FragColor")) return NULL;
    /* sprite atlas position (v_texCoord) as a 2nd output: written in the ID pass to the slot's uv attachment */
    int es3_ = es3 && decl_ls != NULL; const char *uvx = (es3_ && strstr(src, "vec2 v_texCoord;")) ? "v_texCoord" : "vec2(-1.0)";
    char uvdecl[128] = "", uvset[160] = "";
    if (es3_) { snprintf(uvdecl, sizeof uvdecl, "layout(location = 1) out highp vec4 nb_uvo;\n");
                snprintf(uvset, sizeof uvset, " nb_uvo = vec4(%s, 1.0, 1.0);", uvx); }
    char tail[1700]; snprintf(tail, sizeof tail,
        "\n%suniform mediump vec4 nb_idc;\nuniform mediump float nb_athr;\nuniform mediump float nb_cm;\nuniform highp mat4 nb_ivp;\nuniform highp vec4 nb_vpa;\n"
        "void main() { nb_om(); if (nb_idc.a > 0.5) {\n"
        /* coverage of this fragment: opaque 1, alpha blending: alpha, additive (dst ONE): brightness (x alpha if src SRC_ALPHA) */
        "  mediump float cov = nb_cm < 0.5 ? 1.0 : (nb_cm < 1.5 ? %s.a : (max(%s.r, max(%s.g, %s.b)) * (nb_cm > 2.5 ? %s.a : 1.0)));\n"
        "  if (cov < nb_athr) discard;\n"
        "  if (nb_idc.a > 1.5) %s = vec4((1.0 / gl_FragCoord.w - nb_vpa.z) / 1000.0, 0.0, 0.0, 1.0);\n"
        "  else { %s = vec4(nb_idc.rg, clamp(cov, 0.0, 1.0), 1.0);%s } } }\n", uvdecl, outname, outname, outname, outname, outname, outname, outname, uvset);   /* ID: r, g = draw id; b = coverage */
    size_t n = strlen(src); char *o = malloc(n + strlen(tail) + 64); size_t pre = m - src;   /* the tail is ~800 B: size it */
    o[0] = 0;
    if (es3_) {   /* two outputs need explicit locations: the game's output -> location 0 */
        size_t d = decl_ls - src; int has = 0; for (const char *q = decl_ls; *q && *q != ';'; q++) if (!strncmp(q, "layout", 6)) has = 1;
        if (!has) { memcpy(o, src, d); o[d] = 0; strcat(o, "layout(location = 0) "); strncat(o, src + d, pre - d); }
        else { memcpy(o, src, pre); o[pre] = 0; }
    } else { memcpy(o, src, pre); o[pre] = 0; }
    strcat(o, "void nb_om"); strcat(o, m + 9);   /* "void main" -> "void nb_om" */
    strcat(o, tail); return o;
}
static char *id_shader_source(GLuint sh) {   /* the source as GL has it (authoritative; ids get reused) */
    GLint n = 0; glGetShaderiv(sh, GL_SHADER_SOURCE_LENGTH, &n); if (n <= 0) return NULL;
    char *b = malloc(n + 1); glGetShaderSource(sh, n + 1, NULL, b); b[n] = 0; return b;
}
/* after the game linked `prog`: swap in the rewritten fragment shader and relink the game's own program */
static void id_hook_program(GLuint prog) {
    if (prog >= ID_MAXP) return;
    IdProg *P = &g_idp[prog]; free(P->vsrc); free(P->fsrc); memset(P, 0, sizeof *P);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok); if (!ok) return;
    GLuint att[8]; GLsizei na = 0, i; glGetAttachedShaders(prog, 8, &na, att); GLuint fs0 = 0;
    for (i = 0; i < na; i++) { GLint ty = 0; glGetShaderiv(att[i], GL_SHADER_TYPE, &ty);
        if (ty == GL_VERTEX_SHADER) P->vsrc = id_shader_source(att[i]); else if (ty == GL_FRAGMENT_SHADER) { fs0 = att[i]; P->fsrc = id_shader_source(att[i]); } }
    if (!P->vsrc || !P->fsrc || !fs0) return;
    P->vs_h = fnv(P->vsrc); P->fs_h = fnv(P->fsrc);
    char *nsrc = id_frag_src(P->fsrc);
    if (!nsrc) { P->depth_only = !strstr(P->fsrc, "gl_FragColor") && !strstr(P->fsrc, "out "); if (!P->depth_only) { g_id_failed++; fprintf(stderr, "[idpass] prog %u: fragment shader not rewritable\n", prog); } return; }
    int es3 = strstr(nsrc, "nb_uvo") != NULL;
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); const char *s = nsrc; glShaderSource(fs, 1, &s, NULL); glCompileShader(fs); free(nsrc);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024] = ""; glGetShaderInfoLog(fs, sizeof log, NULL, log); fprintf(stderr, "[idpass] prog %u: rewritten fs failed: %.300s\n", prog, log); glDeleteShader(fs); g_id_failed++; return; }
    glDetachShader(prog, fs0); glAttachShader(prog, fs); glLinkProgram(prog); glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {   /* restore the game's program */
        char log[1024] = ""; glGetProgramInfoLog(prog, sizeof log, NULL, log); fprintf(stderr, "[idpass] prog %u: relink failed (%.200s); restored\n", prog, log);
        glDetachShader(prog, fs); glAttachShader(prog, fs0); glLinkProgram(prog); glDeleteShader(fs); g_id_failed++; return;
    }
    glDeleteShader(fs);   /* stays alive while attached */
    P->loc_idc = glGetUniformLocation(prog, "nb_idc"); P->loc_athr = glGetUniformLocation(prog, "nb_athr");
    P->loc_ivp = glGetUniformLocation(prog, "nb_ivp"); P->loc_vpa = glGetUniformLocation(prog, "nb_vpa"); P->loc_cm = glGetUniformLocation(prog, "nb_cm");
    GLint cur = 0; glGetIntegerv(GL_CURRENT_PROGRAM, &cur);
    glUseProgram(prog); glUniform4f(P->loc_idc, 0, 0, 0, 0); glUniform1f(P->loc_athr, 0); glUseProgram((GLuint)cur);
    P->hooked = 1; P->es3 = es3; g_id_hooked++;
}
static void id_slot_fbos(IdSlot *sl, int w, int h) {
    if (sl->id_fbo && sl->w == w && sl->h == h) return;
    if (sl->id_fbo) { glDeleteFramebuffers(1, &sl->id_fbo); glDeleteRenderbuffers(1, &sl->id_col); glDeleteRenderbuffers(1, &sl->id_dep); glDeleteRenderbuffers(1, &sl->uv_col);
                      glDeleteFramebuffers(1, &sl->wp_fbo); glDeleteRenderbuffers(1, &sl->wp_col); glDeleteRenderbuffers(1, &sl->wp_dep); }
    GLint pr = 0, pd = 0, prb = 0; glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr); glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd); glGetIntegerv(GL_RENDERBUFFER_BINDING, &prb);
    glGenRenderbuffers(1, &sl->id_col); glBindRenderbuffer(GL_RENDERBUFFER, sl->id_col); glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
    glGenRenderbuffers(1, &sl->id_dep); glBindRenderbuffer(GL_RENDERBUFFER, sl->id_dep); glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glGenFramebuffers(1, &sl->id_fbo); glBindFramebuffer(GL_FRAMEBUFFER, sl->id_fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, sl->id_col);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, sl->id_dep);
    glGenRenderbuffers(1, &sl->uv_col); glBindRenderbuffer(GL_RENDERBUFFER, sl->uv_col); glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA16F, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, sl->uv_col);
    { GLenum db[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1}; glDrawBuffers(2, db); }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "[idpass] ID framebuffer incomplete\n");
    if (g_wp_on) {
        glGenRenderbuffers(1, &sl->wp_col); glBindRenderbuffer(GL_RENDERBUFFER, sl->wp_col); glRenderbufferStorage(GL_RENDERBUFFER, GL_R32F, w, h);
        glGenRenderbuffers(1, &sl->wp_dep); glBindRenderbuffer(GL_RENDERBUFFER, sl->wp_dep); glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        glGenFramebuffers(1, &sl->wp_fbo); glBindFramebuffer(GL_FRAMEBUFFER, sl->wp_fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, sl->wp_col);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, sl->wp_dep);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "[idpass] world-position framebuffer incomplete: off\n"); g_wp_on = 0; }
    }
    if (getenv("NBGL_ID_RGB")) {   /* a copy of the presented frame per slot (alignment tests in on-demand mode) */
        if (sl->rgb_fbo) { glDeleteFramebuffers(1, &sl->rgb_fbo); glDeleteRenderbuffers(1, &sl->rgb_col); }
        glGenRenderbuffers(1, &sl->rgb_col); glBindRenderbuffer(GL_RENDERBUFFER, sl->rgb_col); glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
        glGenFramebuffers(1, &sl->rgb_fbo); glBindFramebuffer(GL_FRAMEBUFFER, sl->rgb_fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, sl->rgb_col);
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd); glBindRenderbuffer(GL_RENDERBUFFER, prb);
    sl->w = w; sl->h = h;
}
static IdSlot *g_slot;
static void id_fbo(int w, int h) {   /* the current slot's framebuffers become the targets of the replays */
    id_slot_fbos(g_slot, w, h); g_id_fbo = g_slot->id_fbo; g_wp_fbo = g_slot->wp_fbo; g_id_w = w; g_id_h = h;
}
static void id_clear_slot(void) {   /* explicit clear at frame start (the game's own clear is mirrored too) */
    GLfloat cc[4]; GLboolean cm[4], dm, sc; GLint sm = 0; glGetFloatv(GL_COLOR_CLEAR_VALUE, cc); glGetBooleanv(GL_COLOR_WRITEMASK, cm);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &dm); glGetIntegerv(GL_STENCIL_WRITEMASK, &sm); sc = glIsEnabled(GL_SCISSOR_TEST);
    GLfloat cd; GLint cs; glGetFloatv(GL_DEPTH_CLEAR_VALUE, &cd); glGetIntegerv(GL_STENCIL_CLEAR_VALUE, &cs); GLint pd = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd);
    if (sc) glDisable(GL_SCISSOR_TEST); glColorMask(1, 1, 1, 1); glDepthMask(1); glStencilMask(0xff); glClearDepthf(1.0f); glClearStencil(0);
    glClearColor(0, 0, 0, 0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_id_fbo); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    if (g_wp_on) { glClearColor(-99999.0f, 0, 0, 0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_wp_fbo); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd); glClearColor(cc[0], cc[1], cc[2], cc[3]); glColorMask(cm[0], cm[1], cm[2], cm[3]); glDepthMask(dm); glStencilMask(sm);
    glClearDepthf(cd); glClearStencil(cs); if (sc) glEnable(GL_SCISSOR_TEST);
}
static int inv4(const float *m, float *o) {   /* column-major 4x4 inverse (MESA gluInvertMatrix) */
    float inv[16]; int i;
    inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
    float det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12]; if (det == 0) return 0;
    for (i = 0; i < 16; i++) o[i] = inv[i] / det; return 1;
}
static float g_anchor[2], g_wref;
static void id_cam_update(void) {   /* inverse camera + anchor (the ground point at the screen centre) for relative world coordinates */
    inv4(g_id_cam, g_ivp);
    float c0[4] = {0, 0, -1, 1}, c1[4] = {0, 0, 1, 1}, p0[4], p1[4];
    for (int r = 0; r < 4; r++) { p0[r] = p1[r] = 0; for (int k = 0; k < 4; k++) { p0[r] += g_ivp[k * 4 + r] * c0[k]; p1[r] += g_ivp[k * 4 + r] * c1[k]; } }
    for (int r = 0; r < 3; r++) { p0[r] /= p0[3]; p1[r] /= p1[3]; }
    float t = (p1[2] != p0[2]) ? -p0[2] / (p1[2] - p0[2]) : 0; g_anchor[0] = p0[0] + t * (p1[0] - p0[0]); g_anchor[1] = -(p0[1] + t * (p1[1] - p0[1]));
    g_wref = g_id_cam[3] * g_anchor[0] + g_id_cam[7] * (-g_anchor[1]) + g_id_cam[15];   /* w = M[3,:] . (x, -y, 0, 1), column-major */
}
static void id_frame_start(void) {
    g_id_active = g_id_on && (g_ondemand || (S->frames % g_id_every) == 0); g_id_ndraw = 0;
    if (!g_id_active) return;
    g_slot = &g_ring[g_ondemand ? S->frames % ID_RING : 0];
    id_fbo(S->width, S->height);
    if (g_id_have_cam) id_cam_update();
    if (g_ondemand) {
        IdSlot *sl = g_slot; sl->valid = 0; sl->frame = (long)S->frames;
        if (sl->jf) { fclose(sl->jf); sl->jf = NULL; } free(sl->js); sl->js = NULL; sl->jsn = 0;
        sl->jf = open_memstream(&sl->js, &sl->jsn); g_id_json = sl->jf;
        id_clear_slot();
    } else { char fn[512]; snprintf(fn, sizeof fn, "%s/f%07lu_draws.jsonl", g_id_dir, (unsigned long)S->frames); g_id_json = fopen(fn, "w"); }
}
static uint64_t id_hash_blob(const uint8_t *p, size_t n, uint64_t h) {   /* FNV-1a 64 over <= 256 K sampled bytes */
    size_t step = n > 262144 ? n / 262144 : 1;
    for (size_t i = 0; i < n; i += step) { h ^= p[i]; h *= 1099511628211ULL; }
    h ^= (uint64_t)n; h *= 1099511628211ULL; return h ? h : 1;
}
static size_t id_px_bytes(uint32_t fmt, uint32_t type) {   /* bytes per pixel of an uncompressed upload (0 = unknown) */
    int ch = fmt == 0x1908 || fmt == 0x80E1 || fmt == 0x8D99 ? 4 : fmt == 0x1907 || fmt == 0x8D98 ? 3 : fmt == 0x190A || fmt == 0x8227 || fmt == 0x8228 ? 2 :
             fmt == 0x1909 || fmt == 0x1906 || fmt == 0x1903 || fmt == 0x8D94 ? 1 : 0;
    if (type == 0x8033 || type == 0x8034 || type == 0x8363) return 2;
    int b = type == 0x1401 || type == 0x1400 ? 1 : type == 0x140B || type == 0x8D61 || type == 0x1403 || type == 0x1402 ? 2 : type == 0x1406 || type == 0x1405 || type == 0x1404 ? 4 : 0;
    return (size_t)ch * b;
}
static void id_tex_upload(unsigned op, const uint64_t *A, uint8_t *cmd) {
    if ((uint32_t)A[0] != GL_TEXTURE_2D) return;
    GLint t = 0; glGetIntegerv(GL_TEXTURE_BINDING_2D, &t); if (t <= 0 || t >= ID_MAXT) return;
    IdTex *T = &g_idt[t]; uint64_t seed = 1469598103934665603ULL;
    if (op == OP_TS) { T->h = 0; T->w = (int)A[3]; T->ht = (int)A[4]; return; }   /* (target, levels, ifmt, w, h): named by its first sub-upload */
    if (A[1] != 0) return;                                                        /* level 0 only */
    uint64_t off; size_t n;
    if (op == OP_TI) {   /* (target, level, ifmt, w, h, border, format, type, pixels) */
        T->w = (int)A[3]; T->ht = (int)A[4]; off = A[8]; n = (size_t)A[3] * A[4] * id_px_bytes((uint32_t)A[6], (uint32_t)A[7]);
        if (off == NBGL_NULL || !n || off + n > g_id_cmdlen) { T->h = 0; return; }
        T->h = id_hash_blob(cmd + off, n, seed ^ (A[2] * 0x9E3779B97F4A7C15ULL) ^ (A[3] << 32 | A[4]));
    } else if (op == OP_CTI) {   /* (target, level, ifmt, w, h, border, imageSize, data) */
        T->w = (int)A[3]; T->ht = (int)A[4]; off = A[7]; n = (size_t)A[6];
        if (off == NBGL_NULL || !n || off + n > g_id_cmdlen) { T->h = 0; return; }
        T->h = id_hash_blob(cmd + off, n, seed ^ (A[2] * 0x9E3779B97F4A7C15ULL) ^ (A[3] << 32 | A[4]));
    } else if (!T->h) {   /* first sub-upload: (target, level, x, y, w, h, format, type | imageSize, pixels) */
        off = A[8]; n = op == OP_TSI ? (size_t)A[4] * A[5] * id_px_bytes((uint32_t)A[6], (uint32_t)A[7]) : (size_t)A[7];
        if (off == NBGL_NULL || !n || off + n > g_id_cmdlen) return;
        T->h = id_hash_blob(cmd + off, n, seed ^ 0x5bd1e995ULL ^ (A[4] << 32 | A[5]));
    }
}
/* called after nb_dispatch for every command */
static void id_post(unsigned op, const uint64_t *A, uint8_t *cmd, uint8_t *reply, uint64_t R) {
    (void)R;
    if (!g_id_on) return;
    if (op == OP_LINK) { id_hook_program((GLuint)A[0]); return; }
    if (op && (op == OP_TI || op == OP_CTI || op == OP_TS || op == OP_TSI || op == OP_CTSI)) { id_tex_upload(op, A, cmd); return; }
    if (op == OP_PBIN && OP_PBIN) {   /* a program from a binary the game saved (of a hooked program: it has the switch) */
        GLuint prog = (GLuint)A[0]; if (prog >= ID_MAXP) return;
        IdProg *P = &g_idp[prog]; free(P->vsrc); free(P->fsrc); memset(P, 0, sizeof *P);
        GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok); if (!ok) return;
        P->loc_idc = glGetUniformLocation(prog, "nb_idc"); P->loc_athr = glGetUniformLocation(prog, "nb_athr");
        P->loc_ivp = glGetUniformLocation(prog, "nb_ivp"); P->loc_vpa = glGetUniformLocation(prog, "nb_vpa"); P->loc_cm = glGetUniformLocation(prog, "nb_cm");
        if (P->loc_idc >= 0 && P->loc_athr >= 0 && P->loc_cm >= 0 && glGetFragDataLocation(prog, "nb_uvo") == 1) {   /* older rewrites are rejected below */
            P->es3 = 1;   /* binaries of older rewrites (no coverage switch) are rejected below */
            GLint cur = 0; glGetIntegerv(GL_CURRENT_PROGRAM, &cur);
            glUseProgram(prog); glUniform4f(P->loc_idc, 0, 0, 0, 0); glUniform1f(P->loc_athr, 0); glUseProgram((GLuint)cur);
            P->hooked = 1; P->vsrc = strdup("(binary)"); P->fsrc = strdup("(binary)"); g_id_hooked++;
        } else {   /* a binary cached before the hook: make it fail (the game then compiles from source, and the hook applies) */
            GLsizei len = (GLsizei)A[3]; uint8_t *junk = calloc(1, len > 16 ? len : 16);
            glProgramBinary(prog, (GLenum)A[1], junk, len > 16 ? len : 16); free(junk);
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            fprintf(stderr, "[idpass] prog %u from a binary without the ID switch: rejected (link status now %d)\n", prog, ok);
            if (ok) g_id_failed++;
        }
        return;
    }
    if (op == OP_USE) { g_id_cur_prog = (GLuint)A[0]; return; }
    if (op == OP_BINDFB) { if (A[0] == GL_FRAMEBUFFER || A[0] == GL_DRAW_FRAMEBUFFER) g_id_cur_dfb = (GLuint)A[1]; return; }
    if (op == OP_BSD && A[2] == 64 && A[1] == 0 && A[3] != NBGL_NULL) {   /* the camera: the 64-byte perspective matrix upload */
        const float *m = (const float *)(cmd + A[3]);
        if (m[7] != 0.0f && m[11] != 0.0f && m[15] != 0.0f) { memcpy(g_id_cam, m, 64); g_id_have_cam = 1; if (g_id_active) id_cam_update(); }
        return;
    }
    if (!g_id_active || g_id_cur_dfb != 0) return;
    if (op == OP_CLEAR) {
        GLfloat cc[4]; glGetFloatv(GL_COLOR_CLEAR_VALUE, cc); if (!g_id_colour) glClearColor(0, 0, 0, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_id_fbo); glClear((GLbitfield)A[0]);
        if (g_wp_on) { glClearColor(-99999.0f, 0, 0, 0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_wp_fbo); glClear((GLbitfield)A[0]); }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glClearColor(cc[0], cc[1], cc[2], cc[3]); return;
    }
    if (op != OP_DA && op != OP_DE && op != OP_DAI && op != OP_DEI && op != OP_DRE) return;
    GLuint p = g_id_cur_prog; int id = ++g_id_ndraw;
    IdProg *P = p < ID_MAXP ? &g_idp[p] : NULL;
    GLint tex0 = 0, at = 0; glGetIntegerv(GL_ACTIVE_TEXTURE, &at); glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex0); glActiveTexture(at);
    GLboolean blend = glIsEnabled(GL_BLEND), dt = glIsEnabled(GL_DEPTH_TEST); GLint bs = 0, bd = 0, dm = 0, cm[4] = {0}, df = 0;
    glGetIntegerv(GL_BLEND_SRC_RGB, &bs); glGetIntegerv(GL_BLEND_DST_RGB, &bd); glGetIntegerv(GL_DEPTH_WRITEMASK, &dm); glGetIntegerv(GL_DEPTH_FUNC, &df);
    glGetIntegerv(GL_COLOR_WRITEMASK, cm);
    if (g_id_json) fprintf(g_id_json, "{\"id\":%d,\"op\":%u,\"prog\":%u,\"hooked\":%d,\"depth_only\":%d,\"mode\":%lu,\"count\":%ld,\"inst\":%ld,\"tex0\":%d,\"blend\":%d,\"bs\":%d,\"bd\":%d,\"depth\":%d,\"dmask\":%d,\"dfunc\":%d,\"cmask\":%d,\"vs\":%u,\"fs\":%u,\"th\":\"%016llx\",\"tw\":%d,\"tht\":%d}\n",
                           id, op, p, P && P->hooked, P && P->depth_only, (unsigned long)A[0], (long)(op == OP_DRE ? A[3] : A[1]),
                           (long)(op == OP_DAI ? A[3] : op == OP_DEI ? A[4] : 1), tex0, blend, bs, bd, dt, dm, df, cm[0] | cm[1] << 1 | cm[2] << 2 | cm[3] << 3,
                           P ? P->vs_h : 0, P ? P->fs_h : 0, (unsigned long long)(tex0 > 0 && tex0 < ID_MAXT ? g_idt[tex0].h : 0),
                           tex0 > 0 && tex0 < ID_MAXT ? g_idt[tex0].w : 0, tex0 > 0 && tex0 < ID_MAXT ? g_idt[tex0].ht : 0);
    if (!P) return;
    if (P->depth_only) {   /* same depth in the ID buffer for later depth-tested passes */
        GLboolean cmk[4]; glGetBooleanv(GL_COLOR_WRITEMASK, cmk);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_id_fbo); glColorMask(0, 0, 0, 0);
        nb_dispatch(op, A, cmd, reply);
        if (g_wp_on) { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_wp_fbo); nb_dispatch(op, A, cmd, reply); }
        glColorMask(cmk[0], cmk[1], cmk[2], cmk[3]); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        return;
    }
    if (!P->hooked) return;
    if (blend && bd == GL_ONE && !g_id_keep_add && !g_id_colour) return;   /* additive = light only (cannot hide anything): glow is not labelled */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_id_fbo);
    if (!P->es3) { GLenum db[2] = {GL_COLOR_ATTACHMENT0, GL_NONE}; glDrawBuffers(2, db); }
    if (blend && !g_id_colour) glDisable(GL_BLEND);
    glUniform4f(P->loc_idc, (id & 255) / 255.0f, ((id >> 8) & 255) / 255.0f, ((id >> 16) & 255) / 255.0f, g_id_colour ? 0.0f : 1.0f);
    { GLint sd = 0; glGetIntegerv(GL_BLEND_DST_RGB, &sd); float cmode = !blend ? 0.0f : (sd == GL_ONE ? (bs == GL_SRC_ALPHA ? 3.0f : 2.0f) : 1.0f);
      if (P->loc_cm >= 0) glUniform1f(P->loc_cm, cmode); }
    glUniform1f(P->loc_athr, blend ? g_athr : -1.0f);   /* opaque draws may write any alpha: no alpha test; blended: faint
                                                           effects / glows / halos (alpha < NBGL_ID_ATHR, default 0.3) do not claim pixels */
    nb_dispatch(op, A, cmd, reply);
    if (!P->es3) { GLenum db[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1}; glDrawBuffers(2, db); }
    if (g_wp_on && P->loc_vpa >= 0 && !g_id_colour) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_wp_fbo);
        glUniform4f(P->loc_idc, 0, 0, 0, 2.0f); if (P->loc_ivp >= 0) glUniformMatrix4fv(P->loc_ivp, 1, GL_FALSE, g_ivp);
        GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp); glUniform4f(P->loc_vpa, (float)S->width, (float)S->height, g_wref, 0.0f); (void)vp;
        nb_dispatch(op, A, cmd, reply);
    }
    glUniform4f(P->loc_idc, 0, 0, 0, 0); glUniform1f(P->loc_athr, 0); if (P->loc_cm >= 0) glUniform1f(P->loc_cm, 0);
    if (blend && !g_id_colour) glEnable(GL_BLEND);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}
/* before eglSwapBuffers: dump the ID buffer (uint16 ids, rows top-down, zlib; header "NBID" w h) and, with
 * NBGL_ID_RGB=1, the frame (RGB, top-down, zlib, header "NBRG" w h) */
static void id_write_z(const char *fn, const char *magic, int w, int h, const uint8_t *data, size_t n) {
    uLongf zn = compressBound(n); uint8_t *z = malloc(zn);
    if (compress2(z, &zn, data, n, 1) == Z_OK) { FILE *f = fopen(fn, "wb"); if (f) { int32_t hd[2] = {w, h}; fwrite(magic, 1, 4, f); fwrite(hd, 4, 2, f); fwrite(z, 1, zn, f); fclose(f); } }
    free(z);
}
static void id_dump_programs(void);
/* write one slot: ID buffer (full res, uint16), eye depth (x2 down, float32), draws, meta; rgb only for the sampled mode */
static void id_dump_slot(IdSlot *sl, int with_rgb) {
    int w = sl->w, h = sl->h; uint8_t *px = malloc((size_t)w * h * 4); unsigned long fr = (unsigned long)sl->frame;
    GLint pr = 0; glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr); GLint pa = 4; glGetIntegerv(GL_PACK_ALIGNMENT, &pa); glPixelStorei(GL_PACK_ALIGNMENT, 1);
    char fn[512];
    glBindFramebuffer(GL_READ_FRAMEBUFFER, sl->id_fbo); glReadBuffer(GL_COLOR_ATTACHMENT0); glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
    uint16_t *ids = malloc((size_t)w * h * 2);
    for (int y = 0; y < h; y++) { const uint8_t *r = px + (size_t)(h - 1 - y) * w * 4; uint16_t *o = ids + (size_t)y * w;
        for (int x = 0; x < w; x++) o[x] = (uint16_t)(r[4 * x] | r[4 * x + 1] << 8); }
    snprintf(fn, sizeof fn, "%s/f%07lu_id.z", g_id_dir, fr); id_write_z(fn, "NBID", w, h, (uint8_t *)ids, (size_t)w * h * 2); free(ids);
    { uint8_t *cv = malloc((size_t)w * h);   /* coverage of the front-most fragment (0..255) */
      for (int y = 0; y < h; y++) { const uint8_t *r = px + (size_t)(h - 1 - y) * w * 4; uint8_t *o = cv + (size_t)y * w; for (int x = 0; x < w; x++) o[x] = r[4 * x + 2]; }
      snprintf(fn, sizeof fn, "%s/f%07lu_cv.z", g_id_dir, fr); id_write_z(fn, "NBCV", w, h, cv, (size_t)w * h); free(cv); }
    if (sl->uv_col) {   /* sprite atlas position of the front-most fragment */
        int f = 2, ow = w / f, oh = h / f; float *uv = malloc((size_t)w * h * 16); float *o = malloc((size_t)ow * oh * 8);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sl->id_fbo); glReadBuffer(GL_COLOR_ATTACHMENT1); glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, uv);
        for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
            const float *q = uv + ((size_t)(h - 1 - (y * f + f / 2)) * w + (x * f + f / 2)) * 4; int ok = q[3] > 0.5f;
            o[((size_t)y * ow + x) * 2] = ok ? q[0] : -1.0f; o[((size_t)y * ow + x) * 2 + 1] = ok ? q[1] : -1.0f; }
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        snprintf(fn, sizeof fn, "%s/f%07lu_uv.z", g_id_dir, fr); id_write_z(fn, "NBUV", ow, oh, (uint8_t *)o, (size_t)ow * oh * 8);
        free(uv); free(o);
    }
    if (g_wp_on) {
        int f = 2, ow = w / f, oh = h / f; float *wp = malloc((size_t)w * h * 16); float *o = malloc((size_t)ow * oh * 4);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sl->wp_fbo); glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, wp);
        for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) o[(size_t)y * ow + x] = wp[((size_t)(h - 1 - (y * f + f / 2)) * w + (x * f + f / 2)) * 4];
        snprintf(fn, sizeof fn, "%s/f%07lu_wd.z", g_id_dir, fr); id_write_z(fn, "NBWD", ow, oh, (uint8_t *)o, (size_t)ow * oh * 4);
        free(wp); free(o);
    }
    if ((with_rgb || sl->rgb_fbo) && getenv("NBGL_ID_RGB")) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sl->rgb_fbo ? sl->rgb_fbo : 0); glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
        uint8_t *rgb = malloc((size_t)w * h * 3);
        for (int y = 0; y < h; y++) { const uint8_t *r = px + (size_t)(h - 1 - y) * w * 4; uint8_t *o = rgb + (size_t)y * w * 3;
            for (int x = 0; x < w; x++) { o[3 * x] = r[4 * x]; o[3 * x + 1] = r[4 * x + 1]; o[3 * x + 2] = r[4 * x + 2]; } }
        snprintf(fn, sizeof fn, "%s/f%07lu_rgb.z", g_id_dir, fr); id_write_z(fn, "NBRG", w, h, rgb, (size_t)w * h * 3); free(rgb);
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glPixelStorei(GL_PACK_ALIGNMENT, pa); free(px);
    if (sl->js) { snprintf(fn, sizeof fn, "%s/f%07lu_draws.jsonl", g_id_dir, fr); FILE *f = fopen(fn, "w"); if (f) { fwrite(sl->js, 1, sl->jsn, f); fclose(f); } }
    snprintf(fn, sizeof fn, "%s/f%07lu_meta.json", g_id_dir, fr);
    FILE *f = fopen(fn, "w");
    if (f) { fprintf(f, "{\"frame\":%lu,\"w\":%d,\"h\":%d,\"draws\":%d,\"hooked\":%ld,\"failed\":%ld,\"anchor\":[%.3f,%.3f],\"wref\":%.6f,\"cam\":[", fr, w, h, sl->ndraw, g_id_hooked, g_id_failed, sl->anchor[0], sl->anchor[1], sl->wref);
             for (int i = 0; i < 16; i++) fprintf(f, "%s%.9g", i ? "," : "", sl->cam[i]); fprintf(f, "]}\n"); fclose(f); }
    g_id_frame_dumps++;
    if (g_ondemand && (g_id_frame_dumps == 1 || g_id_frame_dumps % 500 == 0)) id_dump_programs();
}
/* serve the collector's requests for frames still in the ring (requests come in increasing frame order) */
static void id_poll_requests(void) {
    if (!g_req) return;
    long newest = -1, oldest = -1;
    for (int k = 0; k < ID_RING; k++) if (g_ring[k].valid) { if (g_ring[k].frame > newest) newest = g_ring[k].frame; if (oldest < 0 || g_ring[k].frame < oldest) oldest = g_ring[k].frame; }
    while (g_req->r < __atomic_load_n(&g_req->w, __ATOMIC_ACQUIRE)) {
        long f = (long)g_req->f[g_req->r % 1000];
        if (f > newest) break;                                       /* not rendered yet: keep it pending */
        IdSlot *sl = NULL; for (int k = 0; k < ID_RING; k++) if (g_ring[k].valid && g_ring[k].frame == f) sl = &g_ring[k];
        if (sl) { id_dump_slot(sl, 0); g_req_served++; } else g_req_late++;
        __atomic_store_n(&g_req->r, g_req->r + 1, __ATOMIC_RELEASE);
    }
    (void)oldest;
}
/* before eglSwapBuffers */
static void id_frame_end(void) {
    if (!g_id_active) return;
    IdSlot *sl = g_slot; if (!sl) return;
    sl->ndraw = g_id_ndraw; memcpy(sl->cam, g_id_cam, 64); sl->anchor[0] = g_anchor[0]; sl->anchor[1] = g_anchor[1]; sl->wref = g_wref;
    if (g_ondemand) {
        if (sl->rgb_fbo) {   /* copy the frame about to be presented (back buffer of FB0) */
            GLint pr = 0, pd = 0; glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &pr); glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &pd);
            GLboolean sc = glIsEnabled(GL_SCISSOR_TEST); if (sc) glDisable(GL_SCISSOR_TEST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, sl->rgb_fbo);
            glBlitFramebuffer(0, 0, sl->w, sl->h, 0, 0, sl->w, sl->h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            if (sc) glEnable(GL_SCISSOR_TEST); glBindFramebuffer(GL_READ_FRAMEBUFFER, pr); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pd);
        }
        if (sl->jf) { fflush(sl->jf); }
        sl->valid = 1; g_id_json = NULL;
        id_poll_requests();
        if ((g_req_served + g_req_late) && ((g_req_served + g_req_late) % 500) == 0 && sl->frame % 100 == 0)
            fprintf(stderr, "[idpass] requests served %ld, too late %ld\n", g_req_served, g_req_late);
    } else {
        sl->frame = (long)S->frames;
        if (g_id_json) { fclose(g_id_json); g_id_json = NULL; }
        char fn[512]; snprintf(fn, sizeof fn, "%s/f%07lu_draws.jsonl", g_id_dir, (unsigned long)S->frames);   /* already on disk */
        free(sl->js); sl->js = NULL; sl->jsn = 0; (void)fn;
        id_dump_slot(sl, 1);
    }
}
/* the shader sources of every linked program (for classifying draws) */
static void id_dump_programs(void) {
    if (!g_id_on) return;
    char fn[512], fnt[520]; snprintf(fn, sizeof fn, "%s/programs.jsonl", g_id_dir); snprintf(fnt, sizeof fnt, "%s.tmp", fn);
    FILE *f = fopen(fnt, "w"); if (!f) return;
    for (int p = 0; p < ID_MAXP; p++) {
        IdProg *P = &g_idp[p]; if (!P->vsrc) continue;
        fprintf(f, "{\"prog\":%d,\"hooked\":%d,\"depth_only\":%d,\"vs\":%u,\"fs\":%u}\n", p, P->hooked, P->depth_only, P->vs_h, P->fs_h);
        char sn[512], tn[520]; FILE *g;   /* written once each (a rewrite during a reader's open looked empty to the labeller) */
        snprintf(sn, sizeof sn, "%s/shader_%u.vs", g_id_dir, P->vs_h);
        if (access(sn, F_OK)) { snprintf(tn, sizeof tn, "%s.tmp", sn); g = fopen(tn, "w"); if (g) { fputs(P->vsrc, g); fclose(g); rename(tn, sn); } }
        snprintf(sn, sizeof sn, "%s/shader_%u.fs", g_id_dir, P->fs_h);
        if (access(sn, F_OK)) { snprintf(tn, sizeof tn, "%s.tmp", sn); g = fopen(tn, "w"); if (g) { fputs(P->fsrc, g); fclose(g); rename(tn, sn); } }
    }
    fclose(f); rename(fnt, fn);
}
