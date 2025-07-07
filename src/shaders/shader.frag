#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normalize(fragNormal), lightDir), 0.0);
    vec3 diffuse = vec3(1.0) * diff; // White color
    vec3 ambient = vec3(0.1);
    outColor = vec4(ambient + diffuse, 1.0);
} 