// struct Uniforms {
//     bounds: vec4<u32>
// };

// @group(0) @binding(0) var<uniform> uniforms: Uniforms;


@group(0) @binding(1) var<storage, read> read_buffer: array<f32>;

/// Output buffer is index 0 because it was easier to program
@group(0) @binding(0) var<storage, read_write> write_buffer: array<f32>;

fn is_outside_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
    return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y) || coord.z >= u32(bounds.z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (is_outside_bounds(global_id, uniforms.bounds.xyz)) {
    // if (is_outside_bounds(global_id, vec3<u32>(8,1,1))) {
    if (global_id.x >= 64) {
        return;
    }

    write_buffer[global_id.x] += read_buffer[global_id.x] * 2.0 + 1.0;
    // write_buffer[global_id.x] += read_buffer[global_id.x] * 2 + 1;
    // write_buffer[global_id.x] = global_id.x +1;
    // write_buffer[global_id.x] = u32(1);
}