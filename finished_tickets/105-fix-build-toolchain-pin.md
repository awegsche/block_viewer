# 105 - fix build: pin toolchain to stable

## Problem

`cargo build` / `cargo check` failed after a toolchain update. The active
default rustup toolchain had become `nightly-x86_64-pc-windows-msvc`
(rustc 1.100.0-nightly, 2026-09-05). Compiling `bevy_ecs 0.15.0` under
that nightly produced dozens of `E0283 type annotations needed` errors
inside `function_system.rs`'s `impl_system_function!` macro expansion
(the `SystemParamFunction::run::call_inner` calls for systems with many
params) — a trait-inference regression relative to what `bevy` 0.15.0
(released Nov 2024) was written against.

The current `stable` toolchain (rustc 1.98.1, 2026-09-01) compiles the
whole workspace cleanly, so this is specific to the nightly channel, not
a real incompatibility with modern rustc in general. (Also confirmed the
sibling `ranvil` crate needs a newer-than-1.85 stable, since it uses
`std::fs::File::try_lock`/`TryLockError`, stabilized after 1.85 — so
pinning to an old stable isn't the right fix either.)

## Fix

Added `rust-toolchain.toml` pinning this project to
`stable-x86_64-pc-windows-msvc`, so `cargo build`/`run`/`check` use
stable here regardless of the user's global rustup default. Verified
`cargo build` and `cargo check --all-targets` both succeed.
