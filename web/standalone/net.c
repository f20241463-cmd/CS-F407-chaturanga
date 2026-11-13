// Same forward pass as netjs.js, compiled to WebAssembly with SIMD.
#define C 64
static float W[835780];
static float IN[64 * 13];
static float OUT[4097];
static float A[64 * C], B[64 * C], T[64 * C];
static float PF[128], VF[64], V1[64];
static short NB[64 * 9];
// offsets of each tensor inside W, in this order:
// in.w in.b, r0a.w r0a.b r0b.w r0b.b ... r3b.b, pc.w pc.b, vc.w vc.b, pf.w pf.b, v1.w v1.b, v2.w v2.b
static int OFF[26];
float *get_w(void) { return W; }
float *get_in(void) { return IN; }
float *get_out(void) { return OUT; }
int *get_off(void) { return OFF; }
void init(void) {
  for (int p = 0; p < 64; p++) for (int k = 0; k < 9; k++) {
    int r = (p >> 3) + k / 3 - 1, c = (p & 7) + k % 3 - 1;
    NB[p * 9 + k] = (r >= 0 && r < 8 && c >= 0 && c < 8) ? r * 8 + c : -1;
  }
}
static void conv3(const float *x, int Ci, const float *w, const float *bias, float *out, const float *resid) {
  for (int p = 0; p < 64; p++) {
    float acc[C];
    for (int co = 0; co < C; co++) acc[co] = bias[co];
    for (int k = 0; k < 9; k++) {
      int q = NB[p * 9 + k];
      if (q < 0) continue;
      const float *xr = x + q * Ci;
      const float *wk = w + k * Ci * C;
      for (int ci = 0; ci < Ci; ci++) {
        float v = xr[ci];
        if (v == 0.0f) continue;
        const float *wr = wk + ci * C;
        for (int co = 0; co < C; co++) acc[co] += v * wr[co];
      }
    }
    float *o = out + p * C;
    if (resid) { const float *rr = resid + p * C; for (int co = 0; co < C; co++) { float s = acc[co] + rr[co]; o[co] = s > 0 ? s : 0; } }
    else for (int co = 0; co < C; co++) o[co] = acc[co] > 0 ? acc[co] : 0;
  }
}
void forward(void) {
  conv3(IN, 13, W + OFF[0], W + OFF[1], A, 0);
  float *h = A, *other = B;
  for (int i = 0; i < 4; i++) {
    conv3(h, 64, W + OFF[2 + 4 * i], W + OFF[3 + 4 * i], T, 0);
    conv3(T, 64, W + OFF[4 + 4 * i], W + OFF[5 + 4 * i], other, h);
    float *s = h; h = other; other = s;
  }
  const float *pw = W + OFF[18], *pb = W + OFF[19], *vw = W + OFF[20], *vb = W + OFF[21];
  for (int p = 0; p < 64; p++) {
    float s0 = pb[0], s1 = pb[1], sv = vb[0];
    const float *o = h + p * C;
    for (int ci = 0; ci < C; ci++) { float v = o[ci]; s0 += v * pw[ci * 2]; s1 += v * pw[ci * 2 + 1]; sv += v * vw[ci]; }
    PF[p * 2] = s0 > 0 ? s0 : 0; PF[p * 2 + 1] = s1 > 0 ? s1 : 0; VF[p] = sv > 0 ? sv : 0;
  }
  const float *fw = W + OFF[22], *fb = W + OFF[23];
  for (int j = 0; j < 4096; j++) OUT[j] = fb[j];
  for (int i = 0; i < 128; i++) { float v = PF[i]; if (v == 0.0f) continue; const float *r = fw + i * 4096; for (int j = 0; j < 4096; j++) OUT[j] += v * r[j]; }
  const float *w1 = W + OFF[24], *b1 = W + OFF[25];
  for (int j = 0; j < 64; j++) V1[j] = b1[j];
  for (int i = 0; i < 64; i++) { float v = VF[i]; if (v == 0.0f) continue; for (int j = 0; j < 64; j++) V1[j] += v * w1[i * 64 + j]; }
  // value head output layer is done in JS (needs tanh); store pre-activation sum of v2 here
  OUT[4096] = 0;
  for (int j = 0; j < 64; j++) V1[j] = V1[j] > 0 ? V1[j] : 0;
}
float *get_v1(void) { return V1; }
