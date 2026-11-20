#version 450
// SPDX-License-Identifier: MIT
// std140 UBO: projection, model, view, plus prevModelMatrix - the SAME object's transform at the PREVIOUS host
// frame. It exists so this host can hand Frame Generation the motion vectors it OWES: a moving scene whose
// motion image is all zeros is a scene FG has nothing to interpolate, and the provider would be interpolating
// a stationary image while the triangle visibly moves. prevModelMatrix is written per frame by the app, so the
// velocity is measured from the app's own transform, not assumed.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(binding = 0) uniform Buffers {
    mat4 projectionMatrix;
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 prevModelMatrix;
} buffers;

layout(location = 0) out vec3 fsColor;
// Consumed ONLY by velocity.frag; the colour pass declares no input at this location, which is legal (an
// unused vertex output). Per-vertex clip-space delta, interpolated by the rasterizer: exactly the velocity
// field for a translating object, and near-exact under this app's slow Z rotation (documented, not hidden).
layout(location = 2) out vec2 velOut;

void main(void) {
    vec4 clip = buffers.projectionMatrix * buffers.viewMatrix * buffers.modelMatrix * vec4(inPos, 1.0);
    vec4 prev = buffers.projectionMatrix * buffers.viewMatrix * buffers.prevModelMatrix * vec4(inPos, 1.0);
    gl_Position = clip;
    fsColor = inColor;
    velOut = clip.xy - prev.xy;
}

