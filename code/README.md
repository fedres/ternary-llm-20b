# TernaryLLM-20B Build Instructions

## Overview
This directory contains the TernaryLLM-20B demonstration program with comprehensive examples of training, inference, evaluation, and benchmarking modes.

## Directory Structure
```
code/
├── include/           # Header files for TernaryLLM components
│   ├── attention_layers.hpp
│   ├── kv_cache.hpp
│   ├── layer_normalization.hpp
│   ├── projections.hpp
│   ├── ternary_core.hpp
│   └── ternary_matrix_ops.hpp
└── src/
    └── main.cpp       # Main demonstration program
```

## Building the Program

### Using g++ (Basic)
```bash
g++ -std=c++17 -O2 -o ternary_llm src/main.cpp -I./include
```

### Using clang++ (Recommended)
```bash
clang++ -std=c++17 -O2 -march=native -o ternary_llm src/main.cpp -I./include
```

### Using Make (if Makefile is created)
```bash
make
```

### Recommended Build Flags
```bash
# Release build with optimizations
g++ -std=c++17 -O3 -DNDEBUG -march=native \
    -o ternary_llm src/main.cpp \
    -I./include \
    -lpthread

# Debug build
g++ -std=c++17 -g -O0 -DDEBUG \
    -o ternary_llm_debug src/main.cpp \
    -I./include
```

## Usage Examples

### 1. Basic Inference
```bash
./ternary_llm --mode inference --model ./ternary_llm_20b.bin --prompt "Hello World"
```

### 2. Performance Benchmarking
```bash
./ternary_llm --mode benchmark --batch-size 64 --seq-length 2048 --benchmark-iters 500
```

### 3. Model Evaluation
```bash
./ternary_llm --mode eval --dataset ./validation_set --output eval_results.json
```

### 4. Training Mode
```bash
./ternary_llm --mode train --config config.json --num-layers 24
```

### 5. Memory-Optimized Inference
```bash
./ternary_llm --mode inference --no-cache --no-quant --batch-size 8
```

### 6. High-Throughput Benchmarking
```bash
./ternary_llm --mode benchmark \
    --batch-size 128 \
    --seq-length 1024 \
    --warmup-iters 100 \
    --benchmark-iters 1000 \
    --output benchmark_results.txt
```

### 7. Custom Model Configuration
```bash
./ternary_llm --mode inference \
    --model ./custom_model.bin \
    --hidden-dim 8192 \
    --num-heads 64 \
    --num-layers 48 \
    --max-seq-len 4096 \
    --prompt "Explain the concept of attention mechanisms in transformers"
```

## Command-Line Options

### Mode Selection
- `--mode train`: Training mode with dummy data
- `--mode inference`: Text generation from prompts
- `--mode eval`: Model evaluation on datasets
- `--mode benchmark`: Performance benchmarking

### Model Configuration
- `--model PATH`: Path to model weights (required)
- `--hidden-dim SIZE`: Hidden dimension size (default: 4096)
- `--num-heads COUNT`: Number of attention heads (default: 32)
- `--num-layers COUNT`: Number of transformer layers (default: 24)

### Inference Parameters
- `--prompt TEXT`: Text prompt for generation
- `--batch-size SIZE`: Batch size for processing (default: 32)
- `--seq-length LEN`: Sequence length (default: 512)
- `--max-seq-len LEN`: Maximum sequence length (default: 2048)

### Performance Options
- `--no-cache`: Disable KV cache
- `--no-quant`: Disable ternary quantization
- `--warmup-iters ITERS`: Warmup iterations (default: 10)
- `--benchmark-iters ITERS`: Benchmark iterations (default: 100)

### Output and Logging
- `--output FILE`: Output file for results
- `--log-level LEVEL`: Logging level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)

### Help
- `--help, -h`: Show help message with all options

## Features Demonstrated

### 1. Ternary Arithmetic
- Bit-packed ternary value operations
- Efficient matrix multiplication using ternary logic
- Performance optimizations through bit-level parallelism

### 2. Attention Mechanisms
- Full attention layer implementation
- KDA (Key-Dot-Attention) for fast inference
- RoPE (Rotary Positional Embedding) support
- Causal masking for autoregressive generation

### 3. Memory Management
- KV cache implementation
- Efficient memory usage tracking
- Quantization support for reduced memory footprint

### 4. Performance Monitoring
- High-resolution timing
- Memory usage measurement
- Throughput calculation
- Comprehensive benchmarking

### 5. Integration Testing
- Model initialization verification
- Forward pass correctness
- Component integration tests
- Error handling validation

## System Requirements

### Minimum Requirements
- C++17 compatible compiler (GCC 8+, Clang 6+, MSVC 2019+)
- 8GB RAM for basic inference
- 1GB disk space for executables and logs

### Recommended Requirements
- 64GB+ RAM for full model inference
- Multi-core CPU (8+ cores) for parallel processing
- SSD storage for faster model loading

### For Benchmarking
- 128GB+ RAM for large batch sizes
- High-performance CPU with AVX2/AVX512 support
- Fast storage for large datasets

## Troubleshooting

### Compilation Errors
1. Ensure C++17 support: `g++ --version` or `clang++ --version`
2. Check include paths: Ensure `-I./include` is specified
3. Verify compiler flags: Use `-std=c++17` minimum

### Runtime Issues
1. Model path: Ensure model file exists and is readable
2. Memory: Check available RAM before running large benchmarks
3. Permissions: Ensure write permissions for output files

### Performance Issues
1. Enable optimizations: Use `-O3` or `-march=native`
2. Increase warmup iterations for more stable benchmarks
3. Disable cache/quantization for debugging: `--no-cache --no-quant`

## Integration with Actual TernaryLLM

This main.cpp file provides a complete framework for the actual TernaryLLM-20B implementation. To integrate with real components:

1. Replace placeholder classes with actual TernaryLLM implementations
2. Include actual header files from the `include/` directory
3. Implement actual model loading and inference logic
4. Add real dataset loading for evaluation mode
5. Implement actual training loops for training mode

The current implementation provides a complete testing and demonstration framework that can be directly adapted for production use.

## Author Information
- **Author**: Zombie
- **Date**: 2025-11-05
- **Version**: 1.0.0
- **License**: See project license

## Support
For issues, questions, or contributions, please refer to the main project repository.