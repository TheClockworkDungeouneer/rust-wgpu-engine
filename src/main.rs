

// use std::{thread::sleep, time::Duration};

// use pollster::block_on;
#[allow(unused_imports)]
use rust_webgpu::{run, run_no_graphics};


fn main() {
    // sleep(Duration::from_millis(5000));
    pollster::block_on(run());
    // pollster::block_on(run_no_graphics());
}
