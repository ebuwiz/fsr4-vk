#version 450
// SPDX-License-Identifier: MIT
// std140 UBO: projection, model, view.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(binding = 0) uniform Buffers {
    mat4 projectionMatrix;
    mat4 modelMatrix;
    mat4 viewMatrix;
} buffers;

layout(location = 0) out vec3 fsColor;

void main(void) {
    gl_Position = buffers.projectionMatrix * buffers.viewMatrix * buffers.modelMatrix * vec4(inPos, 1.0);
    fsColor = inColor;
}

