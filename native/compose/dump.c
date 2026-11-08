/* dump.c (rendclone round, included by replay.c): per-draw dumps of chosen frames, every input resolved from GL itself.
 * Shadows: buffer contents (by this context's buffer name) and per-texture upload info (content hash of the level-0
 * upload, size, format, upload count). For a dumped frame, each draw call writes one JSON line to <out>/f%07d/draws.jsonl:
 *   call (op, mode, count, type, index offset, instances, first), fbo, viewport, scissor, raster state,
 *   program (name, vs / fs hash), uniforms (name -> values), samplers (unit -> texture hash / size / format / params),
 *   uniform blocks (name -> binding, data blob hash), attributes (name -> location, enabled, buffer blob hash, size,
 *   type, normalized, integer, stride, offset, divisor | current value), element buffer blob hash.
 * Blobs (deduplicated by 64-bit content hash) go to <out>/blobs/<hash>.bin. */
#include <sys/stat.h>
#include <zlib.h>
#define SH_MAXB 65536
typedef struct { uint8_t *d; size_t n, cap; } ShBuf;
static ShBuf g_sb[SH_MAXB];
typedef struct { uint64_t h; int w, ht, ifmt, nup, levels, frows; float *fsh; } ShTex;   /* fsh: CPU copy of RGBA32F textures (bones) */
static ShTex g_st[SH_MAXB];
static uint64_t hash64(const uint8_t *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ULL); size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); h = (h ^ v) * 0xff51afd7ed558ccdULL; h ^= h >> 32; }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ULL;
    return h ^ (h >> 29);
}
static long g_frame_now; static int g_bver[SH_MAXB], g_bsaved_ver[SH_MAXB]; static long g_bwframe[SH_MAXB]; static uint64_t g_bsaved_key[SH_MAXB];
static void buf_written(GLuint b) { if (b && b < SH_MAXB) { g_bver[b]++; g_bwframe[b] = g_frame_now; } }
static GLuint bound_buf(GLenum target) {
    GLenum q = 0;
    switch (target) { case GL_ARRAY_BUFFER: q = GL_ARRAY_BUFFER_BINDING; break; case GL_ELEMENT_ARRAY_BUFFER: q = GL_ELEMENT_ARRAY_BUFFER_BINDING; break;
    case GL_UNIFORM_BUFFER: q = GL_UNIFORM_BUFFER_BINDING; break; case GL_COPY_READ_BUFFER: q = GL_COPY_READ_BUFFER_BINDING; break;
    case GL_COPY_WRITE_BUFFER: q = GL_COPY_WRITE_BUFFER_BINDING; break; case GL_PIXEL_UNPACK_BUFFER: q = GL_PIXEL_UNPACK_BUFFER_BINDING; break;
    case GL_PIXEL_PACK_BUFFER: q = GL_PIXEL_PACK_BUFFER_BINDING; break; case GL_TRANSFORM_FEEDBACK_BUFFER: q = GL_TRANSFORM_FEEDBACK_BUFFER_BINDING; break; }
    GLint b = 0; if (q) glGetIntegerv(q, &b); return (GLuint)b;
}
static void sb_resize(ShBuf *s, size_t n) { if (n > s->cap) { s->cap = n; s->d = realloc(s->d, n); } if (n > s->n) memset(s->d + s->n, 0, n - s->n); s->n = n; }
static GLuint bound_tex(GLenum target) {
    GLenum q = target == GL_TEXTURE_2D ? GL_TEXTURE_BINDING_2D : target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_BINDING_CUBE_MAP :
               target == GL_TEXTURE_2D_ARRAY ? GL_TEXTURE_BINDING_2D_ARRAY : target == GL_TEXTURE_3D ? GL_TEXTURE_BINDING_3D : GL_TEXTURE_BINDING_CUBE_MAP;
    GLint b = 0; glGetIntegerv(q, &b); return (GLuint)b;
}
/* before dispatch (args already remapped) */
static void shadow_pre(unsigned op, const uint64_t *A, uint8_t *cmd) {
    if (op == 28) {   /* glBufferData(target, size, data, usage) */
        GLuint b = bound_buf((GLenum)A[0]); if (!b || b >= SH_MAXB) return;
        ShBuf *s = &g_sb[b]; s->n = 0; sb_resize(s, (size_t)A[1]); if (A[2] != NBGL_NULL) memcpy(s->d, cmd + A[2], (size_t)A[1]); buf_written(b);
    } else if (op == 29) {   /* glBufferSubData(target, off, size, data) */
        GLuint b = bound_buf((GLenum)A[0]); if (!b || b >= SH_MAXB || A[3] == NBGL_NULL) return;
        ShBuf *s = &g_sb[b]; if (A[1] + A[2] > s->n) sb_resize(s, A[1] + A[2]); memcpy(s->d + A[1], cmd + A[3], (size_t)A[2]); buf_written(b);
    } else if (op == 219) {   /* glCopyBufferSubData(rt, wt, roff, woff, size) */
        GLuint r = bound_buf((GLenum)A[0]), w = bound_buf((GLenum)A[1]); if (!r || !w || r >= SH_MAXB || w >= SH_MAXB) return;
        if (A[3] + A[4] > g_sb[w].n) sb_resize(&g_sb[w], A[3] + A[4]);
        if (A[2] + A[4] <= g_sb[r].n) memmove(g_sb[w].d + A[3], g_sb[r].d + A[2], (size_t)A[4]); buf_written(w);
    } else if (op == 121 || op == 37 || op == 259 || op == 126 || op == 38 || op == 160 || op == 163) {
        GLenum tgt = (GLenum)A[0]; GLenum bt = (tgt >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && tgt <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) ? GL_TEXTURE_CUBE_MAP : tgt;
        GLuint t = bound_tex(bt); if (!t || t >= SH_MAXB) return;
        ShTex *T = &g_st[t];
        if (op == 259) { T->levels = (int)A[1]; T->ifmt = (int)A[2]; T->w = (int)A[3]; T->ht = (int)A[4]; T->h = 0; T->nup = 0;
            free(T->fsh); T->fsh = NULL; T->frows = 0; if (T->ifmt == GL_RGBA32F) T->fsh = calloc((size_t)T->w * T->ht * 4, 4); return; }
        if (op == 126 && T->fsh && A[8] != NBGL_NULL && (GLenum)A[6] == GL_RGBA && (GLenum)A[7] == GL_FLOAT && A[1] == 0) {   /* float sub-upload into the CPU copy */
            int x0 = (int)A[2], y0 = (int)A[3], w = (int)A[4], h = (int)A[5]; const float *src = (const float *)(cmd + A[8]);
            for (int y = 0; y < h && y0 + y < T->ht; y++) memcpy(T->fsh + ((size_t)(y0 + y) * T->w + x0) * 4, src + (size_t)y * w * 4, (size_t)w * 16);
            if (y0 + h > T->frows) T->frows = y0 + h;
        }
        int level = (int)A[1]; const uint8_t *data = NULL; size_t n = 0;
        if (op == 121) { if (level == 0) { T->w = (int)A[3]; T->ht = (int)A[4]; T->ifmt = (int)A[2]; }
            if (A[8] != NBGL_NULL) { data = cmd + A[8]; n = nb_pix_host((int)A[3], (int)A[4], (GLenum)A[6], (GLenum)A[7]); } }
        else if (op == 37) { if (level == 0) { T->w = (int)A[3]; T->ht = (int)A[4]; T->ifmt = (int)A[2]; } if (A[7] != NBGL_NULL) { data = cmd + A[7]; n = (size_t)A[6]; } }
        else if (op == 126) { if (A[8] != NBGL_NULL) { data = cmd + A[8]; n = nb_pix_host((int)A[4], (int)A[5], (GLenum)A[6], (GLenum)A[7]); } }
        else if (op == 38) { if (A[8] != NBGL_NULL) { data = cmd + A[8]; n = (size_t)A[7]; } }
        if (data && level == 0 && (T->nup == 0 || op == 121 || op == 37)) T->h = hash64(data, n, (uint64_t)T->w << 32 | (uint64_t)T->ht);
        else if (data && level == 0) T->h = hash64(data, n, T->h);   /* later sub-uploads: a new identity (dynamic texture) */
        T->nup++;
    }
}
static size_t nb_pix_host(int w, int h, GLenum format, GLenum type) {
    int nc = (format == GL_RED || format == GL_RED_INTEGER || format == GL_ALPHA || format == GL_LUMINANCE) ? 1 :
             (format == GL_RG || format == GL_RG_INTEGER || format == GL_LUMINANCE_ALPHA) ? 2 : (format == GL_RGB || format == GL_RGB_INTEGER) ? 3 : 4;
    int bpp = (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_5_5_5_1) ? 2 :
              (type == GL_FLOAT || type == GL_INT || type == GL_UNSIGNED_INT) ? 4 * nc : (type == GL_HALF_FLOAT || type == GL_SHORT || type == GL_UNSIGNED_SHORT) ? 2 * nc :
              (type == GL_UNSIGNED_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_24_8) ? 4 : nc;
    GLint al = 4; glGetIntegerv(GL_UNPACK_ALIGNMENT, &al);
    size_t row = ((size_t)w * bpp + al - 1) / al * al; return w > 0 && h > 0 ? row * (h - 1) + (size_t)w * bpp : 0;
}

/* the real content of buffer `buf` [off, off+n) (authoritative; mapping via COPY_READ, binding restored) */
static long g_shadow_bad;
static uint8_t *read_gl_buf(GLuint buf, size_t off, size_t n) {
    GLint prev = 0; glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &prev);
    glBindBuffer(GL_COPY_READ_BUFFER, buf); uint8_t *o = malloc(n ? n : 1);
    void *p = glMapBufferRange(GL_COPY_READ_BUFFER, off, n, GL_MAP_READ_BIT);
    if (p) { memcpy(o, p, n); glUnmapBuffer(GL_COPY_READ_BUFFER); } else memset(o, 0, n);
    glBindBuffer(GL_COPY_READ_BUFFER, prev);
    if (buf < SH_MAXB && off + n <= g_sb[buf].n && memcmp(o, g_sb[buf].d + off, n)) g_shadow_bad++;
    return o;
}
/* a reference to buffer bytes [lo, lo+n): a buffer written this frame -> the range itself (doff 0); otherwise the whole
 * buffer, saved once per content version, and the range's offset in it (static = not written for 30 frames) */
static uint64_t save_blob(const uint8_t *p, size_t n);
static uint64_t buf_ref(GLuint buf, size_t lo, size_t n, size_t *doff) {
    *doff = 0;
    if (!buf || buf >= SH_MAXB || g_bwframe[buf] >= g_frame_now - 30 || !g_sb[buf].n) {   /* written recently: the range only */ uint8_t *d = read_gl_buf(buf, lo, n); uint64_t h = save_blob(d, n); free(d); return h; }
    if (g_bsaved_ver[buf] != g_bver[buf] + 1) { uint8_t *d = read_gl_buf(buf, 0, g_sb[buf].n); g_bsaved_key[buf] = save_blob(d, g_sb[buf].n); free(d); g_bsaved_ver[buf] = g_bver[buf] + 1; }
    *doff = lo; return g_bsaved_key[buf];
}
/* ---- program introspection (cached per program name; reset on link) ---- */
typedef struct { char name[96]; GLenum type; GLint size, loc; } PUni;
typedef struct { char name[96]; GLint binding, size; } PBlk;
typedef struct { char name[96]; GLenum type; GLint loc; } PAtt;
typedef struct { int valid; uint32_t vsh, fsh; int nu, nb, na; PUni *u; PBlk *b; PAtt *a; } PInfo;
static PInfo g_pi[SH_MAXB];
static uint32_t fnv32(const char *s) { uint32_t h = 2166136261u; for (; s && *s; s++) h = (h ^ (uint8_t)*s) * 16777619u; return h; }
static uint64_t g_eth[SH_MAXB];   /* expertbot texture identity per GL texture (see eth_pre) */
static void dump_fbinfo(FILE *f, GLint fb); static char *g_dumpdir; static uint64_t export_texture(GLuint t, GLenum target); static long g_twritten[65536]; static FILE *g_dj; static long g_dframe = -1; static int g_ddraw;
static void save_shader_src(GLuint sh, uint32_t *hout) {
    GLint n = 0; glGetShaderiv(sh, GL_SHADER_SOURCE_LENGTH, &n); if (n <= 0) { *hout = 0; return; }
    char *b = malloc(n + 1); glGetShaderSource(sh, n + 1, NULL, b); b[n] = 0; *hout = fnv32(b);
    GLint ty = 0; glGetShaderiv(sh, GL_SHADER_TYPE, &ty);
    char p[1024]; snprintf(p, sizeof p, "%s/shaders/%u.%s", g_dumpdir, *hout, ty == GL_VERTEX_SHADER ? "vs" : "fs");
    struct stat st; if (stat(p, &st)) { FILE *f = fopen(p, "w"); if (f) { fputs(b, f); fclose(f); } }
    free(b);
}
static PInfo *pinfo(GLuint prog) {
    if (!prog || prog >= SH_MAXB) return NULL;
    PInfo *P = &g_pi[prog]; if (P->valid) return P;
    free(P->u); free(P->b); free(P->a); memset(P, 0, sizeof *P);
    GLuint att[8]; GLsizei na = 0; glGetAttachedShaders(prog, 8, &na, att);
    for (int i = 0; i < na; i++) { GLint ty = 0; glGetShaderiv(att[i], GL_SHADER_TYPE, &ty); if (ty == GL_VERTEX_SHADER) save_shader_src(att[i], &P->vsh); else save_shader_src(att[i], &P->fsh); }
    GLint n = 0; glGetProgramiv(prog, GL_ACTIVE_UNIFORMS, &n); P->u = calloc(n + 1, sizeof(PUni));
    for (int i = 0; i < n; i++) {
        PUni *u = &P->u[P->nu]; GLint blk = -1; GLuint ui = i; glGetActiveUniformsiv(prog, 1, &ui, GL_UNIFORM_BLOCK_INDEX, &blk);
        if (blk >= 0) continue;
        glGetActiveUniform(prog, i, sizeof u->name, NULL, &u->size, &u->type, u->name); u->loc = glGetUniformLocation(prog, u->name); P->nu++;
    }
    glGetProgramiv(prog, GL_ACTIVE_UNIFORM_BLOCKS, &n); P->b = calloc(n + 1, sizeof(PBlk));
    for (int i = 0; i < n; i++) { PBlk *b = &P->b[P->nb++]; glGetActiveUniformBlockName(prog, i, sizeof b->name, NULL, b->name);
        glGetActiveUniformBlockiv(prog, i, GL_UNIFORM_BLOCK_BINDING, &b->binding); glGetActiveUniformBlockiv(prog, i, GL_UNIFORM_BLOCK_DATA_SIZE, &b->size); }
    glGetProgramiv(prog, GL_ACTIVE_ATTRIBUTES, &n); P->a = calloc(n + 1, sizeof(PAtt));
    for (int i = 0; i < n; i++) { PAtt *a = &P->a[P->na++]; GLint sz; glGetActiveAttrib(prog, i, sizeof a->name, NULL, &sz, &a->type, a->name); a->loc = glGetAttribLocation(prog, a->name); }
    P->valid = 1; return P;
}
/* blobs: appended to <dir>/blobs.dat, index lines "<hash> <offset> <len>" in <dir>/blobs.idx (deduplicated in memory) */
static FILE *g_bdat, *g_bidx; static uint64_t *g_bset; static size_t g_bset_cap, g_bset_n; static long g_bdat_off;
static int bset_add(uint64_t h) {   /* 1 if new */
    if (!h) h = 1;
    if (2 * (g_bset_n + 1) > g_bset_cap) { uint64_t *o = g_bset; size_t oc = g_bset_cap; g_bset_cap = oc ? oc * 2 : 1 << 16; g_bset = calloc(g_bset_cap, 8); g_bset_n = 0;
        for (size_t i = 0; i < oc; i++) if (o[i]) { size_t j = (o[i] * 0x9E3779B97F4A7C15ULL) & (g_bset_cap - 1); while (g_bset[j]) j = (j + 1) & (g_bset_cap - 1); g_bset[j] = o[i]; g_bset_n++; } free(o); }
    size_t j = (h * 0x9E3779B97F4A7C15ULL) & (g_bset_cap - 1);
    while (g_bset[j]) { if (g_bset[j] == h) return 0; j = (j + 1) & (g_bset_cap - 1); }
    g_bset[j] = h; g_bset_n++; return 1;
}
static uint64_t save_blob(const uint8_t *p, size_t n) {
    uint64_t h = hash64(p, n, 0x5eed);
    if (!g_bdat) { char path[1024]; snprintf(path, sizeof path, "%s/blobs.dat", g_dumpdir); g_bdat = fopen(path, "ab"); fseek(g_bdat, 0, SEEK_END); g_bdat_off = ftell(g_bdat);
        snprintf(path, sizeof path, "%s/blobs.idx", g_dumpdir);
        FILE *r = fopen(path, "r"); if (r) { unsigned long k, o, l; while (fscanf(r, "%lx %lu %lu", &k, &o, &l) == 3) bset_add(k); fclose(r); }
        g_bidx = fopen(path, "a"); }
    if (bset_add(h)) { fwrite(p, 1, n, g_bdat); fprintf(g_bidx, "%016lx %ld %zu\n", (unsigned long)h, g_bdat_off, n); g_bdat_off += n; }
    return h;
}
static void dump_begin_frame(long frame) {
    char p[1024]; snprintf(p, sizeof p, "%s/f%07ld", g_dumpdir, frame); mkdir(p, 0755);
    strcat(p, "/draws.jsonl.gz"); g_dj = (FILE *)gzopen(p, "wb1"); g_dframe = frame; g_ddraw = 0;
}
static void dump_end_frame(void) { if (g_dj) gzclose((gzFile)g_dj); g_dj = NULL; g_dframe = -1; }
static void jstr(FILE *f, const char *s) { gzputc((gzFile)f, '"'); for (; *s; s++) { if (*s == '"' || *s == '\\') gzputc((gzFile)f, '\\'); gzputc((gzFile)f, *s); } gzputc((gzFile)f, '"'); }
static int is_sampler(GLenum t) { return t == GL_SAMPLER_2D || t == GL_SAMPLER_CUBE || t == GL_SAMPLER_2D_ARRAY || t == GL_SAMPLER_3D || t == GL_SAMPLER_2D_SHADOW || t == GL_INT_SAMPLER_2D || t == GL_UNSIGNED_INT_SAMPLER_2D; }
static int utype_n(GLenum t, int *isint) {
    *isint = 0;
    switch (t) { case GL_FLOAT: return 1; case GL_FLOAT_VEC2: return 2; case GL_FLOAT_VEC3: return 3; case GL_FLOAT_VEC4: return 4;
    case GL_FLOAT_MAT2: return 4; case GL_FLOAT_MAT3: return 9; case GL_FLOAT_MAT4: return 16; case GL_FLOAT_MAT3x4: case GL_FLOAT_MAT4x3: return 12;
    case GL_INT: case GL_BOOL: case GL_UNSIGNED_INT: *isint = 1; return 1; case GL_INT_VEC2: case GL_BOOL_VEC2: *isint = 1; return 2;
    case GL_INT_VEC3: case GL_BOOL_VEC3: *isint = 1; return 3; case GL_INT_VEC4: case GL_BOOL_VEC4: *isint = 1; return 4; }
    *isint = 1; return 1;
}
/* the byte range [lo, hi) of buffer *buf that attribute loc reads in this draw (0 if disabled / no buffer) */
static int attr_range(GLint loc, long vmin, long vmax, long inst, GLuint *buf, size_t *lo, size_t *hi) {
    GLint en = 0, b = 0, sz = 0, ty = 0, st = 0, dv = 0; void *ptr = NULL;
    glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en); if (!en) return 0;
    glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &b); if (b <= 0) return 0;
    glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_SIZE, &sz); glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_TYPE, &ty);
    glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &st); glGetVertexAttribiv(loc, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &dv);
    glGetVertexAttribPointerv(loc, GL_VERTEX_ATTRIB_ARRAY_POINTER, &ptr);
    int esz = (ty == GL_FLOAT || ty == GL_INT || ty == GL_UNSIGNED_INT || ty == GL_FIXED) ? 4 : (ty == GL_SHORT || ty == GL_UNSIGNED_SHORT || ty == GL_HALF_FLOAT) ? 2 :
              (ty == GL_INT_2_10_10_10_REV || ty == GL_UNSIGNED_INT_2_10_10_10_REV) ? 0 : 1;
    size_t el = esz ? (size_t)esz * sz : 4, stride = st ? (size_t)st : el;
    long r0 = dv ? 0 : vmin, r1 = dv ? (inst + dv - 1) / dv - 1 : vmax; if (r1 < r0) r1 = r0;
    *buf = (GLuint)b; *lo = (size_t)(uintptr_t)ptr + r0 * stride; *hi = (size_t)(uintptr_t)ptr + r1 * stride + el; return 1;
}
/* one draw: op + its (remapped) args */
static void dump_draw(unsigned op, const uint64_t *A, uint8_t *cmd) {
    FILE *f = g_dj; if (!f) return;
    GLint prog = 0, vao = 0, fbo = 0, ebo = 0, vp[4], sc[4]; glGetIntegerv(GL_CURRENT_PROGRAM, &prog); glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo); glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo); glGetIntegerv(GL_VIEWPORT, vp); glGetIntegerv(GL_SCISSOR_BOX, sc);
    dump_fbinfo(f, fbo);
    gzprintf((gzFile)f, "{\"i\":%d,\"op\":%u", g_ddraw++, op);
    /* call */
    long mode = (long)A[0], first = 0, count = 0, type = 0, ioff = 0, inst = 1; int client_idx = 0;
    if (op == 56) { first = (long)A[1]; count = (long)A[2]; }
    else if (op == 226) { first = (long)A[1]; count = (long)A[2]; inst = (long)A[3]; }
    else if (op == 57) { count = (long)A[1]; type = (long)A[2]; ioff = (long)A[3]; client_idx = A[4] != 0; }
    else if (op == 227) { count = (long)A[1]; type = (long)A[2]; ioff = (long)A[3]; inst = (long)A[4]; client_idx = A[5] != 0; }
    else if (op == 159) { count = (long)A[3]; type = (long)A[4]; ioff = (long)A[5]; client_idx = A[6] != 0; }
    gzprintf((gzFile)f, ",\"mode\":%ld,\"first\":%ld,\"count\":%ld,\"itype\":%ld,\"ioff\":%ld,\"inst\":%ld,\"fbo\":%d,\"vp\":[%d,%d,%d,%d]", mode, first, count, type, client_idx ? 0 : ioff, inst, fbo, vp[0], vp[1], vp[2], vp[3]);
    long vmin = first, vmax = first + count - 1;
    if (type) {   /* index data: the used range (client-side indices or the element buffer) */
        size_t isz = type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4;
        const uint8_t *ip = NULL;
        uint8_t *ipm = NULL;
        if (client_idx) ip = cmd + ioff; else if (ebo > 0) ip = ipm = read_gl_buf(ebo, ioff, count * isz);
        if (ip) {
            size_t idoff = 0; uint64_t h = client_idx ? save_blob(ip, count * isz) : buf_ref(ebo, ioff, count * isz, &idoff); uint32_t mn = 0xffffffffu, mx = 0;
            for (long k = 0; k < count; k++) { uint32_t v = isz == 1 ? ip[k] : isz == 2 ? ((const uint16_t *)ip)[k] : ((const uint32_t *)ip)[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
            vmin = mn; vmax = mx;
            gzprintf((gzFile)f, ",\"idx\":\"%016lx\",\"idoff\":%zu,\"imin\":%u,\"imax\":%u,\"ebo\":%d", (unsigned long)h, idoff, mn, mx, client_idx ? -1 : ebo);
            free(ipm);
        }
    }
    /* raster state */
    GLint bsr, bdr, bsa, bda, ber, bea, df, cf, ff, sf, sr, sm, sfail, szf, szp, swm; GLboolean dm, cm[4];
    glGetIntegerv(GL_BLEND_SRC_RGB, &bsr); glGetIntegerv(GL_BLEND_DST_RGB, &bdr); glGetIntegerv(GL_BLEND_SRC_ALPHA, &bsa); glGetIntegerv(GL_BLEND_DST_ALPHA, &bda);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &ber); glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &bea); glGetIntegerv(GL_DEPTH_FUNC, &df); glGetBooleanv(GL_DEPTH_WRITEMASK, &dm);
    glGetBooleanv(GL_COLOR_WRITEMASK, cm); glGetIntegerv(GL_CULL_FACE_MODE, &cf); glGetIntegerv(GL_FRONT_FACE, &ff);
    glGetIntegerv(GL_STENCIL_FUNC, &sf); glGetIntegerv(GL_STENCIL_REF, &sr); glGetIntegerv(GL_STENCIL_VALUE_MASK, &sm); glGetIntegerv(GL_STENCIL_FAIL, &sfail);
    glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &szf); glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &szp); glGetIntegerv(GL_STENCIL_WRITEMASK, &swm);
    GLfloat dr[2], po[2]; glGetFloatv(GL_DEPTH_RANGE, dr); glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &po[0]); glGetFloatv(GL_POLYGON_OFFSET_UNITS, &po[1]);
    gzprintf((gzFile)f, ",\"blend\":[%d,%d,%d,%d,%d,%d,%d],\"depth\":[%d,%d,%d,%g,%g],\"cmask\":[%d,%d,%d,%d],\"cull\":[%d,%d,%d],\"stencil\":[%d,%d,%d,%d,%d,%d,%d,%d],"
            "\"scissor\":[%d,%d,%d,%d,%d],\"poff\":[%d,%g,%g]",
            glIsEnabled(GL_BLEND), bsr, bdr, bsa, bda, ber, bea, glIsEnabled(GL_DEPTH_TEST), df, dm, dr[0], dr[1], cm[0], cm[1], cm[2], cm[3],
            glIsEnabled(GL_CULL_FACE), cf, ff, glIsEnabled(GL_STENCIL_TEST), sf, sr, sm, sfail, szf, szp, swm, glIsEnabled(GL_SCISSOR_TEST), sc[0], sc[1], sc[2], sc[3],
            glIsEnabled(GL_POLYGON_OFFSET_FILL), po[0], po[1]);
    PInfo *P = pinfo((GLuint)prog);
    if (!P) { gzprintf((gzFile)f, "}\n"); return; }
    gzprintf((gzFile)f, ",\"prog\":%d,\"vs\":%u,\"fs\":%u,\"u\":{", prog, P->vsh, P->fsh);
    GLint curunit = 0; glGetIntegerv(GL_ACTIVE_TEXTURE, &curunit); int first_u = 1;
    for (int i = 0; i < P->nu; i++) {
        PUni *u = &P->u[i]; if (u->loc < 0) continue;
        if (!first_u) gzputc((gzFile)f, ','); first_u = 0; jstr(f, u->name); gzputc((gzFile)f, ':');
        if (is_sampler(u->type)) {
            GLint unit = 0; glGetUniformiv(prog, u->loc, &unit);
            GLenum tt = u->type == GL_SAMPLER_CUBE ? GL_TEXTURE_CUBE_MAP : u->type == GL_SAMPLER_2D_ARRAY ? GL_TEXTURE_2D_ARRAY : u->type == GL_SAMPLER_3D ? GL_TEXTURE_3D : GL_TEXTURE_2D;
            glActiveTexture(GL_TEXTURE0 + unit); GLuint t = bound_tex(tt); GLint smp = 0; glGetIntegerv(GL_SAMPLER_BINDING, &smp);
            GLint mn = 0, mg = 0, ws = 0, wt = 0;
            if (smp) { glGetSamplerParameteriv(smp, GL_TEXTURE_MIN_FILTER, &mn); glGetSamplerParameteriv(smp, GL_TEXTURE_MAG_FILTER, &mg); glGetSamplerParameteriv(smp, GL_TEXTURE_WRAP_S, &ws); glGetSamplerParameteriv(smp, GL_TEXTURE_WRAP_T, &wt); }
            else if (t) { glGetTexParameteriv(tt, GL_TEXTURE_MIN_FILTER, &mn); glGetTexParameteriv(tt, GL_TEXTURE_MAG_FILTER, &mg); glGetTexParameteriv(tt, GL_TEXTURE_WRAP_S, &ws); glGetTexParameteriv(tt, GL_TEXTURE_WRAP_T, &wt); }
            ShTex *T = t < SH_MAXB ? &g_st[t] : NULL; (void)0;
            int isrt = t && t < SH_MAXB && g_twritten[t] == g_dframe; uint64_t xh = 0;
            if (t && !isrt && T && T->fsh && T->frows) { isrt = 2; xh = save_blob((const uint8_t *)T->fsh, (size_t)T->w * T->frows * 16); }   /* dynamic float data (bones): rows as a blob */
            else if (t && !isrt) xh = export_texture(t, tt);
            gzprintf((gzFile)f, "{\"rt\":%d,\"x\":\"%016lx\",\"unit\":%d,", isrt, (unsigned long)xh, unit);
            gzprintf((gzFile)f, "\"tex\":%u,\"target\":%u,\"th\":\"%016lx\",\"w\":%d,\"h\":%d,\"fmt\":%d,\"nup\":%d,\"smp\":[%d,%d,%d,%d]}", t, tt,
                    (unsigned long)(T ? T->h : 0), T ? T->w : 0, T ? T->ht : 0, T ? T->ifmt : 0, T ? T->nup : 0, mn, mg, ws, wt);
        } else {
            int isint, nn = utype_n(u->type, &isint) * (u->size > 0 ? u->size : 1); if (nn > 256) nn = 256;
            gzputc((gzFile)f, '[');
            if (isint) { GLint v[256]; glGetUniformiv(prog, u->loc, v); for (int k = 0; k < (u->size > 1 ? 1 : nn); k++) gzprintf((gzFile)f, "%s%d", k ? "," : "", v[k]); }
            else { GLfloat v[256] = {0}; int per = utype_n(u->type, &isint);
                for (int e = 0; e < (u->size > 0 ? u->size : 1) && e * per < 256; e++) { char nm[128]; snprintf(nm, sizeof nm, "%.*s[%d]", (int)(strchr(u->name, '[') ? strchr(u->name, '[') - u->name : (long)strlen(u->name)), u->name, e);
                    GLint l = u->size > 1 ? glGetUniformLocation(prog, nm) : u->loc; if (l >= 0) glGetUniformfv(prog, l, v + e * per); }
                for (int k = 0; k < nn; k++) gzprintf((gzFile)f, "%s%.9g", k ? "," : "", v[k]); }
            gzputc((gzFile)f, ']');
        }
    }
    glActiveTexture(curunit);
    gzprintf((gzFile)f, "},\"ub\":{");
    for (int i = 0; i < P->nb; i++) {
        PBlk *b = &P->b[i]; GLint buf = 0; GLint64 st = 0, sz = 0;
        glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, b->binding, &buf); glGetInteger64i_v(GL_UNIFORM_BUFFER_START, b->binding, &st); glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, b->binding, &sz);
        size_t n = (size_t)b->size; uint64_t h = 0;
        if (buf > 0) { uint8_t *d = read_gl_buf(buf, st, n); h = save_blob(d, n); free(d); }
        gzprintf((gzFile)f, "%s", i ? "," : ""); jstr(f, b->name); gzprintf((gzFile)f, ":{\"binding\":%d,\"buf\":%d,\"off\":%ld,\"size\":%d,\"data\":\"%016lx\"}", b->binding, buf, (long)st, b->size, (unsigned long)h);
    }
    gzprintf((gzFile)f, "},\"a\":{");
    /* one saved range per vertex buffer (the union of its attributes' ranges): interleaved attributes share it */
    GLuint ubuf[16]; size_t ulo[16], uhi[16], udoff[16]; uint64_t ukey[16]; int nu = 0;
    for (int i = 0; i < P->na; i++) {
        GLuint b; size_t lo, hi; if (P->a[i].loc < 0 || !attr_range(P->a[i].loc, vmin, vmax, inst, &b, &lo, &hi)) continue;
        int k = 0; while (k < nu && ubuf[k] != b) k++;
        if (k == nu) { if (nu == 16) continue; ubuf[nu] = b; ulo[nu] = lo; uhi[nu] = hi; nu++; }
        else { if (lo < ulo[k]) ulo[k] = lo; if (hi > uhi[k]) uhi[k] = hi; }
    }
    for (int k = 0; k < nu; k++) ukey[k] = buf_ref(ubuf[k], ulo[k], uhi[k] - ulo[k], &udoff[k]);
    for (int i = 0; i < P->na; i++) {
        PAtt *a = &P->a[i]; if (a->loc < 0) continue;
        GLint en = 0, buf = 0, sz = 0, ty = 0, nm = 0, st = 0, in = 0, dv = 0; void *ptr = NULL;
        glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en); glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buf);
        glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_SIZE, &sz); glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_TYPE, &ty);
        glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &nm); glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &st);
        glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &in); glGetVertexAttribiv(a->loc, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &dv);
        glGetVertexAttribPointerv(a->loc, GL_VERTEX_ATTRIB_ARRAY_POINTER, &ptr);
        if (i) gzputc((gzFile)f, ','); jstr(f, a->name);
        if (en) {
            /* the rows this draw reads: vertices vmin..vmax (divisor 0) or instances 0..inst/div-1 */
            int esz = (ty == GL_FLOAT || ty == GL_INT || ty == GL_UNSIGNED_INT || ty == GL_FIXED) ? 4 : (ty == GL_SHORT || ty == GL_UNSIGNED_SHORT || ty == GL_HALF_FLOAT) ? 2 :
                      (ty == GL_INT_2_10_10_10_REV || ty == GL_UNSIGNED_INT_2_10_10_10_REV) ? 0 : 1;
            size_t el = esz ? (size_t)esz * sz : 4, stride = st ? (size_t)st : el;
            long r0 = dv ? 0 : vmin, r1 = dv ? (inst + dv - 1) / dv - 1 : vmax; if (r1 < r0) r1 = r0;
            uint64_t h = 0; size_t lo = (size_t)(uintptr_t)ptr + r0 * stride, hi = (size_t)(uintptr_t)ptr + r1 * stride + el;
            size_t doff = 0;
            if (buf > 0) { int k = 0; while (k < nu && ubuf[k] != (GLuint)buf) k++;
                if (k < nu) { h = ukey[k]; doff = udoff[k] + (lo - ulo[k]); } else h = buf_ref(buf, lo, hi - lo, &doff); }
            gzprintf((gzFile)f, ":{\"loc\":%d,\"buf\":%d,\"doff\":%zu,\"data\":\"%016lx\",\"row0\":%ld,\"rows\":%ld,\"size\":%d,\"type\":%d,\"norm\":%d,\"int\":%d,\"stride\":%zu,\"off\":%ld,\"div\":%d}",
                    a->loc, buf, doff, (unsigned long)h, r0, r1 - r0 + 1, sz, ty, nm, in, stride, (long)(uintptr_t)ptr, dv);
        } else { GLfloat v[4]; glGetVertexAttribfv(a->loc, GL_CURRENT_VERTEX_ATTRIB, v); gzprintf((gzFile)f, ":{\"loc\":%d,\"cur\":[%g,%g,%g,%g]}", a->loc, v[0], v[1], v[2], v[3]); }
    }
    gzprintf((gzFile)f, "}}\n");
}

/* ---- texture export (shared second context, so the game's GL state is untouched) ----
 * Each texture has a version (bumped by uploads and by being drawn into). A draw that samples a texture written in the
 * same frame references it as a render target ("rt"); otherwise the texture's current content is exported once per
 * version to <out>/tex/<hash>.tex: "NBTX", u32 w, h, levels, export ifmt (RGBA8 / RGBA32F / R32F), original ifmt,
 * then each level's texels (rows bottom-up, as GL stores them). */
static EGLDisplay g_xdpy; static EGLContext g_xctx, g_xmain; static EGLSurface g_xsurf, g_xmsurf;
static int g_tver[SH_MAXB], g_texp_ver[SH_MAXB]; static uint64_t g_texp_h[SH_MAXB];
static GLuint g_fbo_att[SH_MAXB][4];   /* colour 0 / depth / stencil / depth-stencil texture of each FBO (by name) */
static GLuint g_xprog[3], g_xfbo, g_xvao;
static int fmt_class(int ifmt) {   /* 0 rgba8, 1 float, 2 depth */
    switch (ifmt) { case GL_RGBA32F: case GL_RGBA16F: case GL_RGB32F: case GL_RGB16F: case GL_R32F: case GL_R16F: case GL_RG32F: case GL_RG16F: case GL_R11F_G11F_B10F: return 1;
    case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32F: case GL_DEPTH24_STENCIL8: case GL_DEPTH32F_STENCIL8: return 2; }
    return 0;
}
static GLuint xprog(int cls) {
    if (g_xprog[cls]) return g_xprog[cls];
    const char *vs = "#version 300 es\nvoid main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
    const char *fs = cls == 2 ? "#version 300 es\nprecision highp float; uniform highp sampler2D t; uniform int lv; out highp vec4 o; void main(){ o = vec4(texelFetch(t, ivec2(gl_FragCoord.xy), lv).r, 0.0, 0.0, 1.0); }"
                              : "#version 300 es\nprecision highp float; uniform highp sampler2D t; uniform int lv; out highp vec4 o; void main(){ o = texelFetch(t, ivec2(gl_FragCoord.xy), lv); }";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER), p = glCreateProgram();
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v); glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p); return g_xprog[cls] = p;
}
static uint64_t export_texture(GLuint t, GLenum target) {
    if (t >= SH_MAXB) return 0;
    if (g_texp_ver[t] == g_tver[t] + 1) return g_texp_h[t];   /* stored as ver+1 (0 = never) */
    if (target != GL_TEXTURE_2D) return 0;
    eglMakeCurrent(g_xdpy, g_xsurf, g_xsurf, g_xctx);
    if (!g_xfbo) { glGenFramebuffers(1, &g_xfbo); glGenVertexArrays(1, &g_xvao); }
    glBindTexture(GL_TEXTURE_2D, t);
    GLint ifmt = 0, w0 = 0, h0 = 0; glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &ifmt);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w0); glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h0);
    if (g_st[t].ifmt) ifmt = g_st[t].ifmt;
    int cls = fmt_class(ifmt); GLenum efmt = cls == 0 ? GL_RGBA8 : cls == 1 ? GL_RGBA32F : GL_R32F; size_t bpp = cls == 0 ? 4 : cls == 1 ? 16 : 4;
    GLint bmax = 1000, bbase = 0, imm = 0; glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, &bmax); glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, &bbase);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
    GLint cmp = 0; glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, &cmp); if (cmp) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    size_t tot = 0; int nl = 0; int lw[16], lh[16];
    for (int l = 0; l < 16 && l <= bmax; l++) { GLint w = 0, h = 0; glGetTexLevelParameteriv(GL_TEXTURE_2D, l, GL_TEXTURE_WIDTH, &w); glGetTexLevelParameteriv(GL_TEXTURE_2D, l, GL_TEXTURE_HEIGHT, &h);
        if (w <= 0 || h <= 0) break; lw[nl] = w; lh[nl] = h; nl++; tot += (size_t)w * h * bpp; }
    uint64_t hh = 0;
    if (nl > 0) {
        uint32_t hdr[8] = {0x5854424e, (uint32_t)lw[0], (uint32_t)lh[0], (uint32_t)nl, efmt, (uint32_t)ifmt, (uint32_t)(imm ? 1 : 0) | (uint32_t)bbase << 8, (uint32_t)bmax};
        uint8_t *buf = malloc(32 + tot); memcpy(buf, hdr, 32); size_t pos = 32;
        GLuint p = xprog(cls); glUseProgram(p); glUniform1i(glGetUniformLocation(p, "t"), 0); glActiveTexture(GL_TEXTURE0); glBindVertexArray(g_xvao);
        glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST); glDisable(GL_CULL_FACE);
        for (int l = 0; l < nl; l++) {
            GLuint c; glGenTextures(1, &c); glBindTexture(GL_TEXTURE_2D, c); glTexStorage2D(GL_TEXTURE_2D, 1, efmt, lw[l], lh[l]);
            glBindFramebuffer(GL_FRAMEBUFFER, g_xfbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, c, 0);
            glBindTexture(GL_TEXTURE_2D, t); glUniform1i(glGetUniformLocation(p, "lv"), l);
            glViewport(0, 0, lw[l], lh[l]); glDrawArrays(GL_TRIANGLES, 0, 3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            if (cls == 0) glReadPixels(0, 0, lw[l], lh[l], GL_RGBA, GL_UNSIGNED_BYTE, buf + pos);
            else if (cls == 1) glReadPixels(0, 0, lw[l], lh[l], GL_RGBA, GL_FLOAT, buf + pos);
            else { float *tmp = malloc((size_t)lw[l] * lh[l] * 16); glReadPixels(0, 0, lw[l], lh[l], GL_RGBA, GL_FLOAT, tmp);
                   for (size_t k = 0; k < (size_t)lw[l] * lh[l]; k++) memcpy(buf + pos + 4 * k, tmp + 4 * k, 4); free(tmp); }
            pos += (size_t)lw[l] * lh[l] * bpp;
            glBindFramebuffer(GL_FRAMEBUFFER, 0); glDeleteTextures(1, &c);
        }
        if (cmp) { glBindTexture(GL_TEXTURE_2D, t); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, cmp); }
        hh = hash64(buf + 32, tot, 0x7e7);
        char path[1024]; snprintf(path, sizeof path, "%s/tex/%016lx.tex.zst", g_dumpdir, (unsigned long)hh);
        if (t < SH_MAXB && g_eth[t]) { char tp[1024]; snprintf(tp, sizeof tp, "%s/texth.txt", g_dumpdir); FILE *tf = fopen(tp, "a");
            if (tf) { fprintf(tf, "%016lx %016llx %d %d\n", (unsigned long)hh, (unsigned long long)g_eth[t], lw[0], lh[0]); fclose(tf); } }
        struct stat st; if (stat(path, &st)) {
            size_t cap = ZSTD_compressBound(32 + tot); void *z = malloc(cap); size_t zn = ZSTD_compress(z, cap, buf, 32 + tot, 3);
            FILE *f = fopen(path, "wb"); if (f && !ZSTD_isError(zn)) { fwrite(z, 1, zn, f); } if (f) fclose(f); free(z); }
        free(buf);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    eglMakeCurrent(g_xdpy, g_xmsurf, g_xmsurf, g_xmain);
    g_texp_ver[t] = g_tver[t] + 1; g_texp_h[t] = hh; return hh;
}
/* expertbot's texture identity ("th" of glhost_id/idpass.c, the key of its sprite_cells.json): FNV-1a 64 over <= 256 K sampled
 * bytes of the level-0 upload (or the first sub-upload after glTexStorage2D / a NULL upload), seeded with format and size.
 * Tracked per GL texture so every exported texture version can be listed with it (<dump>/texth.txt: key th w h). */
static uint64_t g_eth[SH_MAXB]; static unsigned g_eop_ti, g_eop_cti, g_eop_ts, g_eop_tsi, g_eop_ctsi; static int g_eop_init;
static uint64_t eth_hash(const uint8_t *p, size_t n, uint64_t h) {
    size_t step = n > 262144 ? n / 262144 : 1;
    for (size_t i = 0; i < n; i += step) { h ^= p[i]; h *= 1099511628211ULL; }
    h ^= (uint64_t)n; h *= 1099511628211ULL; return h ? h : 1;
}
static size_t eth_px_bytes(uint32_t fmt, uint32_t type) {
    int ch = fmt == 0x1908 || fmt == 0x80E1 || fmt == 0x8D99 ? 4 : fmt == 0x1907 || fmt == 0x8D98 ? 3 : fmt == 0x190A || fmt == 0x8227 || fmt == 0x8228 ? 2 :
             fmt == 0x1909 || fmt == 0x1906 || fmt == 0x1903 || fmt == 0x8D94 ? 1 : 0;
    if (type == 0x8033 || type == 0x8034 || type == 0x8363) return 2;
    int b = type == 0x1401 || type == 0x1400 ? 1 : type == 0x140B || type == 0x8D61 || type == 0x1403 || type == 0x1402 ? 2 : type == 0x1406 || type == 0x1405 || type == 0x1404 ? 4 : 0;
    return (size_t)ch * b;
}
static void eth_pre(unsigned op, const uint64_t *A, const uint8_t *cmd, size_t cmdlen) {
    if (!g_eop_init) { g_eop_init = 1;
        for (unsigned o = 16; o < 400; o++) { const char *n = nb_op_name(o); if (!n) continue;
            if (!strcmp(n, "glTexImage2D")) g_eop_ti = o; else if (!strcmp(n, "glCompressedTexImage2D")) g_eop_cti = o;
            else if (!strcmp(n, "glTexStorage2D")) g_eop_ts = o; else if (!strcmp(n, "glTexSubImage2D")) g_eop_tsi = o;
            else if (!strcmp(n, "glCompressedTexSubImage2D")) g_eop_ctsi = o; } }
    if (op != g_eop_ti && op != g_eop_cti && op != g_eop_ts && op != g_eop_tsi && op != g_eop_ctsi) return;
    if ((uint32_t)A[0] != GL_TEXTURE_2D) return;
    GLint t = 0; glGetIntegerv(GL_TEXTURE_BINDING_2D, &t); if (t <= 0 || t >= SH_MAXB) return;
    uint64_t seed = 1469598103934665603ULL, off; size_t n;
    if (op == g_eop_ts) { g_eth[t] = 0; return; }
    if (A[1] != 0) return;
    if (op == g_eop_ti) {
        off = A[8]; n = (size_t)A[3] * A[4] * eth_px_bytes((uint32_t)A[6], (uint32_t)A[7]);
        if (off == NBGL_NULL || !n || off + n > cmdlen) { g_eth[t] = 0; return; }
        g_eth[t] = eth_hash(cmd + off, n, seed ^ (A[2] * 0x9E3779B97F4A7C15ULL) ^ (A[3] << 32 | A[4]));
    } else if (op == g_eop_cti) {
        off = A[7]; n = (size_t)A[6];
        if (off == NBGL_NULL || !n || off + n > cmdlen) { g_eth[t] = 0; return; }
        g_eth[t] = eth_hash(cmd + off, n, seed ^ (A[2] * 0x9E3779B97F4A7C15ULL) ^ (A[3] << 32 | A[4]));
    } else if (!g_eth[t]) {
        off = A[8]; n = op == g_eop_tsi ? (size_t)A[4] * A[5] * eth_px_bytes((uint32_t)A[6], (uint32_t)A[7]) : (size_t)A[7];
        if (off == NBGL_NULL || !n || off + n > cmdlen) return;
        g_eth[t] = eth_hash(cmd + off, n, seed ^ 0x5bd1e995ULL ^ (A[4] << 32 | A[5]));
    }
}
/* version bookkeeping, called for every command (args remapped) */
static void texver_pre(unsigned op, const uint64_t *A) {
    if (op == 121 || op == 37 || op == 259 || op == 126 || op == 38 || op == 160 || op == 161 || op == 163 || op == 164 || op == 66 || op == 39 || op == 40) {
        GLenum tgt = (GLenum)A[0]; GLenum bt = (tgt >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && tgt <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) ? GL_TEXTURE_CUBE_MAP : tgt;
        GLuint t = bound_tex(bt); if (t && t < SH_MAXB) g_tver[t]++;
    } else if (op == 63) {   /* glFramebufferTexture2D(target, attachment, textarget, texture, level) */
        GLenum tg = (GLenum)A[0]; GLint fb = 0; glGetIntegerv(tg == GL_READ_FRAMEBUFFER ? GL_READ_FRAMEBUFFER_BINDING : GL_DRAW_FRAMEBUFFER_BINDING, &fb);
        if (fb > 0 && fb < SH_MAXB) { int k = A[1] == GL_COLOR_ATTACHMENT0 ? 0 : A[1] == GL_DEPTH_ATTACHMENT ? 1 : A[1] == GL_STENCIL_ATTACHMENT ? 2 : A[1] == GL_DEPTH_STENCIL_ATTACHMENT ? 3 : -1;
            if (k >= 0) g_fbo_att[fb][k] = (GLuint)A[3]; }
    }
}
/* a draw / clear / blit into the bound draw framebuffer: its textures change */
static void texver_draw(long frame) {
    GLint fb = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb);
    if (fb > 0 && fb < SH_MAXB) for (int k = 0; k < 4; k++) { GLuint t = g_fbo_att[fb][k]; if (t && t < SH_MAXB) { g_tver[t]++; g_twritten[t] = frame; } }
}
/* clears (glClear 31, glClearBuffer* 214..217) and framebuffer descriptions, as their own JSON lines */
static int g_fbdesc_done[SH_MAXB]; static long g_fbdesc_frame[SH_MAXB];
static void dump_fbinfo(FILE *f, GLint fb) {
    if (fb <= 0 || fb >= SH_MAXB || g_fbdesc_frame[fb] == g_dframe) return;
    g_fbdesc_frame[fb] = g_dframe;
    gzprintf((gzFile)f, "{\"fbdesc\":%d", fb);
    GLenum att[3] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT}; const char *an[3] = {"color", "depth", "stencil"};
    for (int k = 0; k < 3; k++) {
        GLint ty = 0, nm = 0; glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, att[k], GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &ty);
        if (ty == GL_NONE) continue;
        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, att[k], GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &nm);
        GLint w = 0, h = 0, fmt = 0;
        if (ty == GL_TEXTURE && nm < SH_MAXB) { w = g_st[nm].w; h = g_st[nm].ht; fmt = g_st[nm].ifmt; }
        else if (ty == GL_RENDERBUFFER) { GLint prb = 0; glGetIntegerv(GL_RENDERBUFFER_BINDING, &prb); glBindRenderbuffer(GL_RENDERBUFFER, nm);
            glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &w); glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &h);
            glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &fmt); glBindRenderbuffer(GL_RENDERBUFFER, prb); }
        gzprintf((gzFile)f, ",\"%s\":{\"type\":%d,\"name\":%d,\"w\":%d,\"h\":%d,\"fmt\":%d}", an[k], ty, nm, w, h, fmt);
    }
    gzprintf((gzFile)f, "}\n");
}
static void dump_clear(unsigned op, const uint64_t *A, uint8_t *cmd) {
    FILE *f = g_dj; if (!f) return;
    GLint fb = 0, vp[4], sc[4], swm = 0; GLfloat cc[4], cd = 1; GLint cs = 0; GLboolean cm[4], dm;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fb); dump_fbinfo(f, fb);
    glGetIntegerv(GL_VIEWPORT, vp); glGetIntegerv(GL_SCISSOR_BOX, sc); glGetFloatv(GL_COLOR_CLEAR_VALUE, cc); glGetFloatv(GL_DEPTH_CLEAR_VALUE, &cd);
    glGetIntegerv(GL_STENCIL_CLEAR_VALUE, &cs); glGetBooleanv(GL_COLOR_WRITEMASK, cm); glGetBooleanv(GL_DEPTH_WRITEMASK, &dm); glGetIntegerv(GL_STENCIL_WRITEMASK, &swm);
    gzprintf((gzFile)f, "{\"clear\":%u,\"fbo\":%d,\"scissor\":[%d,%d,%d,%d,%d],\"cmask\":[%d,%d,%d,%d],\"dmask\":%d,\"smask\":%d", op, fb, glIsEnabled(GL_SCISSOR_TEST), sc[0], sc[1], sc[2], sc[3], cm[0], cm[1], cm[2], cm[3], dm, swm);
    if (op == 31) gzprintf((gzFile)f, ",\"mask\":%lu,\"color\":[%g,%g,%g,%g],\"d\":%g,\"s\":%d", (unsigned long)A[0], cc[0], cc[1], cc[2], cc[3], cd, cs);
    else {   /* glClearBuffer{iv,uiv,fv}(buffer, drawbuffer, value*) / fi(buffer, drawbuffer, depth, stencil) */
        gzprintf((gzFile)f, ",\"buffer\":%lu,\"drawbuffer\":%ld", (unsigned long)A[0], (long)A[1]);
        if (op == 217) gzprintf((gzFile)f, ",\"d\":%g,\"s\":%ld", nb_u2f(A[2]), (long)A[3]);
        else if (A[2] != NBGL_NULL) { const uint32_t *v = (const uint32_t *)(cmd + A[2]); int n = A[0] == GL_COLOR ? 4 : 1;
            gzprintf((gzFile)f, ",\"v\":["); for (int k = 0; k < n; k++) { if (op == 216) { float x; memcpy(&x, v + k, 4); gzprintf((gzFile)f, "%s%g", k ? "," : "", x); } else gzprintf((gzFile)f, "%s%d", k ? "," : "", (int)v[k]); } gzprintf((gzFile)f, "]"); }
    }
    gzprintf((gzFile)f, "}\n");
}
