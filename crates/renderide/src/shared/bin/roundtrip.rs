//! Roundtrip binary for C#–Rust serialization tests.
//! Usage: roundtrip <type_name> <input.bin> <output.bin>
//! Reads bytes from input, unpacks the type, packs to output.

use std::env;
use std::fs;
use std::process;

use renderide::shared;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: roundtrip <type_name> <input.bin> <output.bin>");
        process::exit(1);
    }
    let type_name = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    let input = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", input_path, e);
        process::exit(1);
    });

    let output = shared::roundtrip_dispatch(type_name, &input).unwrap_or_else(|e| {
        eprintln!("Roundtrip failed for {}: {}", type_name, e);
        process::exit(1);
    });

    fs::write(output_path, &output).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {}", output_path, e);
        process::exit(1);
    });
}
