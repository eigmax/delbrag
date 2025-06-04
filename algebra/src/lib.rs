#![allow(dead_code)]
use wrk17::Gate;

mod bigint;
mod bitwise;
mod u32;

pub struct GatePropogator {
    gate_index: u64,
    gates: Vec<Gate>,
}
