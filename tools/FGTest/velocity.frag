#version 450
// SPDX-License-Identifier: MIT
// MOTION-VECTOR PASS: writes the interpolated per-vertex clip-space delta from triangle.vert into an
// R16G16_SFLOAT image. The pass CLEARS the attachment to (0,0) first, so pixels the triangle does NOT cover
// keep ZERO velocity - which is correct: the background is static in this app. Only inside the silhouette is
// the velocity non-zero, exactly like a real game's G-buffer.
// NO UBO is declared here: the value arrives as a varying, so this fragment stage cannot disagree with the
// vertex stage's block layout.
layout(location = 2) in vec2 velIn;
layout(location = 0) out vec2 outVel;

void main(void) {
    outVel = velIn;
}
