// use std::thread::sleep;
// use std::time::Duration;
#![allow(unused_imports)]
use std::{/*intrinsics::nontemporal_store*/ iter, num::NonZero};
use bytemuck::{self, bytes_of};
// use futures_intrusive::channel::shared::oneshot_channel;

// Graphics
use wgpu::{core::device, util::DeviceExt};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};


#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;


use wgpu::Color;
use winit::window::Window;

// #[allow(dead_code)]
struct Graphics<'a> {
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,    
    render_pipeline: wgpu::RenderPipeline,
    // vertex_buffer: wgpu::Buffer, 
    objects: Vec<Object>,
    clear_color: wgpu::Color,
}

pub struct Compute {
    compute_pipeline: wgpu::ComputePipeline,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    map_buffer: Option<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
}
impl Compute {
    pub fn calculate(self: &mut Self, device: &wgpu::Device, queue: &wgpu::Queue) {
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        { 
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.dispatch_workgroups(1, 1, 1);
            // compute_pass.dispatch_workgroups(self.input_buffer.size() as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

    }

    pub fn write_buffer(self, queue: &wgpu::Queue, /*encoder: wgpu::CommandEncoder,*/ data: Vec<u32>) {
        // let staging_buffer = queue.write_buffer_with(&self.input_buffer, 0, NonZero::<u64>::new(self.input_buffer.size()).unwrap()).unwrap();
        // staging_buffer.
        queue.write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(data.as_slice()));
    }
    
    pub async fn read_buffer(self, queue: &wgpu::Queue, device: &wgpu::Device) -> Vec<u32> {
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Read Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.output_buffer, 0, &self.map_buffer.as_ref().unwrap(), 0, self.output_buffer.size()
        );

        queue.submit(std::iter::once(encoder.finish()));


        let map_buff = self.map_buffer.as_ref().unwrap();

        let result;
        {
            let buffer_slice = map_buff.slice(..);

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |i| {
                tx.send(i).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();
            
            let data = buffer_slice.get_mapped_range();
            
            result = bytemuck::cast_slice(data.to_vec().as_slice()).to_vec(); //.unwrap();

        }    

        map_buff.unmap();
    
        
        return result;
    }
    
    fn init_buffers(device: &wgpu::Device, size: NonZero<u32>) -> (wgpu::Buffer, wgpu::Buffer, Option<wgpu::Buffer>) {
        // let data: [u8; size] = core::array::from_fn(|i| i as u8);
        let data: Vec<u32> = Vec::from_iter(0..(size.get() as u32));

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some(&"Default Compute Input"),
            contents: bytemuck::cast_slice(data.as_slice()), //bytemuck::cast_slice(&[0..16]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,//MAP_WRITE,
        });
        // let input_buffer = device.create_buffer(&wgpu::BufferDescriptor{
        //     label: Some(&"Default Compute Input"),
        //     size,
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE,
        //     mapped_at_creation: false,
        // });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&"Default Compute Input"),
            size: (size.get() as usize * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let map_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&"Default Compute Mapping Buffer"),
            size: (size.get() as usize * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false
        });


        return (input_buffer, output_buffer, Some(map_buffer));

    }

    pub fn new(device: &wgpu::Device, size: NonZero<u32>) -> Self {

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some(&"Compute Bind Group Layout"),
            entries: &[
                // Input Buffer
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None },
                    count: None //Some(size),
                },

                wgpu::BindGroupLayoutEntry{
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None },
                    count: None //Some(size),

                },
            ]
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: Some(&"Default Compute Layout"), 
            bind_group_layouts: &[&bind_group_layout], 
            push_constant_ranges: &[] });
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Default Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into())
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&"Default Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let (input_buffer, output_buffer, map_buffer) = Self::init_buffers(device, size);        
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&"Default Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: input_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: output_buffer.as_entire_binding()
                },
            ]
        });



        return Self{compute_pipeline,input_buffer, output_buffer, map_buffer, bind_group};
    }
    
    // fn init_pipeline(self:&mut Self, device: &wgpu::Device) {   
    // }
}

struct Object {
    vertex_buffer: wgpu::Buffer,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];

struct State<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    graphics: Graphics<'a>,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();
        
        // Ensure non-zero size
        // while size.width == 0 || size.height == 0 {
        //     // Wait for the next frame to give the browser time to update the DOM
        //     // window.request_redraw();
        //     // You could optionally wait for a small period here if needed
        //     // tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        //     size = window.inner_size();
        // }

        log::warn!("test: Size: width:{} | height: {}", size.width, size.height);

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch="wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch="wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY, //wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                memory_hints: Default::default(),
            },
            None, // Trace path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into())
        });
    
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{ 
            label: Some("Main render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()], 
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })], 
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                // cull_mode: Some(wgpu::Face::Back),
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None,
            cache: None 
        });
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let objects = Vec::from([Object{vertex_buffer}]);

        Self {
            device,
            queue,
            graphics: Graphics{ 
                surface,
                config,
                size,
                window,
                render_pipeline,
                objects,
                clear_color: Color{r:0.0, g:0.0, b:0.0, a:1.0}
            }
        }
    }

    pub fn window(&self) -> &Window {
        &self.graphics.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.graphics.size = new_size;
            self.graphics.config.width = new_size.width;
            self.graphics.config.height = new_size.height;
            self.graphics.surface.configure(&self.device, &self.graphics.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event{
        WindowEvent::CursorMoved {position, .. } => {
            self.graphics.clear_color = Color {
                r: position.x / self.graphics.size.width as f64,
                b: position.y / self.graphics.size.height as f64,
                g: (position.x / self.graphics.size.width as f64) * (position.y / self.graphics.size.height as f64),
                a:1.0
            };
            true
        }
        _ => false
        }
    }

    fn update(&mut self) {
    
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.graphics.surface.get_current_texture()?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });


        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // load: wgpu::LoadOp::Clear(wgpu::Color {
                        //     r: 0.1,
                        //     g: 0.2,
                        //     b: 0.3,
                        //     a: 1.0,
                        // }),
                        load: wgpu::LoadOp::Clear(self.graphics.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.graphics.render_pipeline);
            render_pass.set_vertex_buffer(0, self.graphics.objects[0].vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..1);
            // render_pass.draw_indexed();
        }


        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

} 


#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    // Start logger. If compiling to wasm, then pipe to the JS console.
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
            // console_log::init_with_level(log::Level::Debug).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
 
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        
        log::warn!("working?");
        
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");

        let _ = window.request_inner_size(PhysicalSize::new(450, 400));
        // event_loop.run()
        // window.request_redraw();
    }
    

    let mut state = State::new(&window).await;
    let mut surface_configured = false;

    let mut compute: Compute = Compute::new(&state.device, NonZero::new(16).unwrap());
    compute.calculate(&state.device, &state.queue);
    
    let test = compute.read_buffer(&state.queue, &state.device).await;
    
    println!("Compute: {:?}", test);

    event_loop.run(move |event, control_flow| {
        // state.resize(state.graphics.size);
        // state.surface.configure(&state.device, &state.config);
        // log::warn!("mid test: Width: {} | height: {}", state.graphics.size.width, state.graphics.size.height);        


        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            // What would happen if I set it to happen on a different window's ID?
            } if window_id == state.window().id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => control_flow.exit(),
                    WindowEvent::Resized(physical_size) => {
                        log::info!("physical_size: {physical_size:?}");
                        surface_configured = true;
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        // This tells winit that we want another frame after this one
                        state.window().request_redraw();

                        if !surface_configured {
                            return;
                        }

                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if it's lost or outdated
                            Err(
                                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                            ) => state.resize(state.graphics.size),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OutOfMemory");
                                control_flow.exit();
                            }

                            // This happens when the a frame takes too long to present
                            Err(wgpu::SurfaceError::Timeout) => {
                                log::warn!("Surface timeout")
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        // log::warn!("End: Width: {} | height: {}", state.graphics.size.width, state.graphics.size.height);
    }).unwrap();
    
}