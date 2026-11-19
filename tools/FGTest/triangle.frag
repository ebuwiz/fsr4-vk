#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 fsColor;
layout(location = 0) out vec4 outColor;

void main(void) {
    outColor = vec4(fsColor, 1.0);
}
