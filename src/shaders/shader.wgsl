
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    // @builtin(vertex_index) in_vertex_index: u32,
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    // let x = f32(1 - i32(in_vertex_index)) * 0.5;
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    let x = model.position.x;
    let y = model.position.y;
    
    out.clip_position = vec4<f32>(model.position.xyz, 1.0);
    out.position = vec2<f32>(model.position.xy);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(
    // @builtin(front_facing) front_facing: bool,
    in: VertexOutput
) -> @location(0) vec4<f32> {
    // return vec4<f32>(in.clip_position.x / 500.0, in.clip_position.y / 500.0, 0.1, 1.0);
    return vec4<f32>(in.color, 1.0);
    // return vec4<f32>(in.position.x, in.position.y, 0.1, 1.0);
}
