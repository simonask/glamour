# Run `make -f rebuild-wasmtime-guest.make` to rebuild `tests/wasmtime-guests.wasm`.

.PHONY: all
all: tests/wasmtime_guest.wasm

target/wasm32-wasip2/release/wasmtime_guest.wasm: tests/wasmtime-guest/src/lib.rs tests/wasmtime-guest/wit/world.wit
	cargo build --release --target=wasm32-wasip2 -p wasmtime-guest

tests/wasmtime_guest.wasm: target/wasm32-wasip2/release/wasmtime_guest.wasm
	cp $< $@

