BIN=./target/debug/beam
TARGET=Beam.app/Contents/MacOS/Beam

$(BIN): src/*.rs
	cargo build

Beam.app: $(BIN)
	mkdir -p Beam.app/Contents/MacOS
	cp $(BIN) $(TARGET)
	cp -r assets Beam.app/Contents/MacOS

clean:
	rm -rf Beam.app
	rm -rf pack

app: Beam.app

pack/beam: $(BIN)
	mkdir -p pack/assets
	mkdir -p pack/shaders
	cp $(BIN) pack/beam
	cp -r assets pack
	cp -r shaders pack

.PHONY: clean app

beam: $(BIN)
	cp $(BIN) beam
