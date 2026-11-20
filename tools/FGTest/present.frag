// present pass: samples the AMD-FSR upscaled FP16 output and overlays a top-right
// FPS number (5x3 bitmap digits, value via push constants). Never fakes anything:
// it only draws what the upscaler actually produced.
#version 450
layout(binding = 0) uniform sampler2D srcTex;
layout(push_constant) uniform PC {
    uint fps;    // integer frames-per-second, 0..999
    uint W;      // presentation width
    uint H;      // presentation height
    uint cell;   // pixels per bitmap cell for the glyphs
} pc;
layout(location = 0) out vec4 o;

int mask(int d) {
    if (d == 0) return 31699; if (d == 1) return 9362;  if (d == 2) return 31015; if (d == 3) return 31207;
    if (d == 4) return 18924; if (d == 5) return 31183; if (d == 6) return 31215; if (d == 7) return 2343;
    if (d == 8) return 31727; return 2543; // 9
}
bool lit(int d, int r, int c) { return ((mask(d) >> (r * 3 + c)) & 1) != 0; }

void main() {
    vec2 t = vec2(gl_FragCoord.x / float(pc.W), 1.0 - gl_FragCoord.y / float(pc.H));
    vec4 col = texture(srcTex, t);
    float cellf = float(pc.cell);
    uint fps = pc.fps;
    uint nd = (fps >= 100u) ? 3u : (fps >= 10u) ? 2u : 1u;
    float boxRight  = float(pc.W) - cellf;              // 1-cell right margin
    float boxLeft   = boxRight - float(nd * 3u) * cellf;
    float boxTop    = cellf;                             // 1-cell top margin
    float boxBottom = boxTop + 5.0 * cellf;
    if (gl_FragCoord.x >= boxLeft && gl_FragCoord.x < boxRight &&
        gl_FragCoord.y >= boxTop  && gl_FragCoord.y <  boxBottom) {
        float relx = gl_FragCoord.x - boxLeft;
        float rely = gl_FragCoord.y - boxTop;
        uint cix = uint(relx / cellf);        // 0..(nd*3-1)
        uint dcol = cix / 3u;                 // which digit from the left
        int c = int(cix % 3u);
        int r = int(rely / cellf);
        uint divisor = 1u;
        for (uint k = 0u; k < nd - 1u - dcol; ++k) divisor *= 10u;
        int d = int((fps / divisor) % 10u);
        if (lit(d, r, c)) col = vec4(1.0, 1.0, 1.0, 1.0);
    }
    o = col;
}
