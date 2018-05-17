layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat4 Model;
    mat3 TBN;
} gs_in[];

out GS_OUT {
    vec3 Normal;
    vec3 FragPos;
    vec2 TexCoords;
    vec3 OriginalPos;
    mat3 TBN;
} gs_out;

uniform float time;

void emit(int i, mat3 TBN) {
    gl_Position = gl_in[i].gl_Position;
    gs_out.Normal = gs_in[i].Normal;
    gs_out.FragPos = gs_in[i].FragPos;
    gs_out.OriginalPos = gs_in[i].OriginalPos;
    gs_out.TBN = TBN;
    gs_out.TexCoords = gs_in[i].TexCoords;
    EmitVertex();
}

void main() {
    for (int i = 0; i < 3; i++) {
        // int a = i;
        // int b = (i + 1) % 3;
        // int c = (i + 2) % 3;

        // vec3 pos1 = gs_in[a].OriginalPos.xyz;
        // vec3 pos2 = gs_in[b].OriginalPos.xyz;
        // vec3 pos3 = gs_in[c].OriginalPos.xyz;
        // vec2 uv1 = gs_in[a].TexCoords;
        // vec2 uv2 = gs_in[b].TexCoords;
        // vec2 uv3 = gs_in[c].TexCoords;

        // vec3 edge1 = pos2 - pos1;
        // vec3 edge2 = pos3 - pos1;
        // vec2 deltaUV1 = uv2 - uv1;
        // vec2 deltaUV2 = uv3 - uv1;

        // float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

        // vec3 tangent = f * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);

        // vec3 T = normalize(vec3(gs_in[a].Model * vec4(tangent, 0.0)));
        // vec3 N = normalize(vec3(gs_in[a].Model * vec4(gs_in[a].Normal, 0.0)));

        // T = normalize(T - dot(T, N) * N);

        // vec3 B = cross(N, T);

        // mat3 TBN = mat3(T, B, N);
        mat3 TBN = gs_in[i].TBN;

        emit(i, TBN);
    }

    // Non working

    // vec3 dXYZdU = gs_in[1].OriginalPos.xyz - gs_in[0].OriginalPos.xyz;
    // float dSdU = gs_in[1].TexCoords.s - gs_in[0].TexCoords.s;

    // vec3 dXYZdV = gs_in[2].OriginalPos.xyz - gs_in[0].OriginalPos.xyz;
    // float dSdV = gs_in[2].TexCoords.s - gs_in[0].TexCoords.s;

    // vec3 tangent = normalize(dSdV * dXYZdU - dSdU * dXYZdV);
    // for (int i = 0; i < 3; i++) {
    //     vec3 normal = gs_in[i].Normal;
    //     vec3 bitangent = cross(-tangent, normal);

    //     // vec3 T = normalize(vec3(gs_in[i].Model * vec4(tangent, 0.0)));
    //     // vec3 N = normalize(vec3(gs_in[i].Model * vec4(gs_in[i].Normal, 0.0)));

    //     // T = normalize(T - dot(T, N) * N);

    //     // vec3 B = cross(N, T);

    //     // mat3 TBN = mat3(T, B, N);

    //     mat3 TBN = mat3(tangent, bitangent, normal);

    //     emit(i, TBN);
    // }
}
