# Run `make -f rebuild-wasmtime-guest.make` to rebuild `tests/wasmtime-guests.wasm`.

.PHONY: all
all: tests/wasmtime_guest.wasm

wasi_snapshot_preview1.proxy.wasm:
	curl -OLsS http://github.com/bytecodealliance/wasmtime/releases/download/v24.0.0/wasi_snapshot_preview1.proxy.wasm

target/wasm32-wasip1/release/wasmtime_guest.wasm: tests/wasmtime-guest/src/lib.rs
	cargo build --release --target=wasm32-wasip1 -p wasmtime-guest

tests/wasmtime_guest.wasm: target/wasm32-wasip1/release/wasmtime_guest.wasm wasi_snapshot_preview1.proxy.wasm
	wasm-tools component new $< --adapt ./wasi_snapshot_preview1.proxy.wasm -o $@

