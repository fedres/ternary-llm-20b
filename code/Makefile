# TernaryLLM-20B Makefile
# Author: Zombie
# Date: 2025-11-05
# 
# This Makefile provides convenient build targets for the TernaryLLM-20B demonstration

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPT_FLAGS = -O2
DEBUG_FLAGS = -g -O0 -DDEBUG
INCLUDES = -I./include

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Files
MAIN_SRC = $(SRC_DIR)/main.cpp
TARGET = $(BIN_DIR)/ternary_llm
TARGET_DEBUG = $(BIN_DIR)/ternary_llm_debug

# Create directories
DIRS = $(BUILD_DIR) $(BIN_DIR)

# Default target
.DEFAULT_GOAL := all

# Create directories if they don't exist
$(DIRS):
	@mkdir -p $@

# Help target
help:
	@echo "TernaryLLM-20B Build System"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build optimized release version (default)"
	@echo "  debug       - Build debug version"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Build and run with default parameters"
	@echo "  test        - Build and run integration tests"
	@echo "  benchmark   - Build and run performance benchmarks"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make all                    # Build release version"
	@echo "  make debug                  # Build debug version"
	@echo "  make run MODE=inference     # Build and run inference"
	@echo "  make benchmark BATCH=64     # Build and run benchmark"

# Build optimized release version
all: $(TARGET)
	@echo "Build complete: $(TARGET)"

$(TARGET): $(MAIN_SRC) | $(BIN_DIR)
	@echo "Building optimized release version..."
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(INCLUDES) -o $@ $<
	@echo "Created: $(TARGET)"

# Build debug version
debug: $(TARGET_DEBUG)
	@echo "Build complete: $(TARGET_DEBUG)"

$(TARGET_DEBUG): $(MAIN_SRC) | $(BIN_DIR)
	@echo "Building debug version..."
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) -o $@ $<
	@echo "Created: $(TARGET_DEBUG)"

# Run with default parameters
run: $(TARGET)
	@echo "Running TernaryLLM-20B in inference mode..."
	@./$(TARGET) --mode inference --model ./model.bin --prompt "Hello, TernaryLLM!"

# Run integration tests
test: $(TARGET)
	@echo "Running integration tests..."
	@./$(TARGET) --mode inference --model ./model.bin --log-level 2

# Run performance benchmark
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@./$(TARGET) --mode benchmark --batch-size 32 --seq-length 512 --benchmark-iters 100

# Run with custom parameters
# Usage: make run MODE=train BATCH=64 SEQ=1024
run:
	@if [ -z "$(MODE)" ]; then \
		echo "Error: MODE parameter required"; \
		echo "Usage: make run MODE=<train|inference|eval|benchmark>"; \
		exit 1; \
	fi
	$(TARGET) --mode $(MODE) --model ./model.bin $(if $(BATCH),--batch-size $(BATCH)) $(if $(SEQ),--seq-length $(SEQ))

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Install (if applicable)
install: $(TARGET)
	@echo "Installing TernaryLLM-20B..."
	@sudo cp $(TARGET) /usr/local/bin/ || echo "Installation requires sudo privileges"
	@echo "Installation complete"

# Uninstall (if applicable)
uninstall:
	@echo "Uninstalling TernaryLLM-20B..."
	@sudo rm -f /usr/local/bin/ternary_llm || echo "Uninstall requires sudo privileges"
	@echo "Uninstall complete"

# Check dependencies
check:
	@echo "Checking dependencies..."
	@echo "C++ Compiler:"
	@$(CXX) --version || echo "Error: C++ compiler not found"
	@echo "C++17 support:"
	@echo 'int main() { return 0; }' | $(CXX) -x c++ -std=c++17 - -o /dev/null 2>&1 && echo "OK" || echo "Not available"

# Package for distribution
package: clean all
	@echo "Creating distribution package..."
	@mkdir -p package/ternary_llm
	@cp $(TARGET) package/ternary_llm/
	@cp README.md package/ternary_llm/
	@cp -r include package/ternary_llm/
	@tar -czf ternary_llm_20b_demo.tar.gz package/ternary_llm
	@rm -rf package
	@echo "Package created: ternary_llm_20b_demo.tar.gz"

# Print variables for debugging
debug-vars:
	@echo "Build Configuration:"
	@echo "  CXX: $(CXX)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  OPT_FLAGS: $(OPT_FLAGS)"
	@echo "  DEBUG_FLAGS: $(DEBUG_FLAGS)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  MAIN_SRC: $(MAIN_SRC)"
	@echo "  TARGET: $(TARGET)"
	@echo "  TARGET_DEBUG: $(TARGET_DEBUG)"

# Phony targets
.PHONY: all debug clean run test benchmark run check package install uninstall help debug-vars