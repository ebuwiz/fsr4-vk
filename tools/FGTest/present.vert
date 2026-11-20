// fullscreen-triangle present pass for --window (renders the AMD-FSR upscaled FP16 target
// to the swapchain and draws the FPS readout top-right).
#version 450
layout(location = 0) out vec2 in_uv;
void main() {
    vec2 p = vec2(gl_VertexIndex == 2 ? 3.0 : -1.0, gl_VertexIndex == 1 ? 3.0 : -1.0);
    in_uv = p * 0.5; // [-1,1] -> [-1,1] uv; flipped in the fragment shader
    gl_Position = vec4(p, 0.0, 1.0);
}
