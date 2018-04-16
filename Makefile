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

app: Beam.app

.PHONY: clean app
