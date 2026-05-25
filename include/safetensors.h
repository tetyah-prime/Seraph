// safetensors.h - SafeTensors format reader
// Zero-copy memory-mapped weight loading for Safetensor model files
// Part of TETYAH-PRIME's native training and inference engine

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

// Data types supported by safetensors
typedef enum {
    DTYPE_UNKNOWN = 0,
    DTYPE_F32,      // float32
    DTYPE_F16,      // float16
    DTYPE_BF16,     // bfloat16
    DTYPE_I8,       // int8
    DTYPE_I16,      // int16
    DTYPE_I32,      // int32
    DTYPE_I64,      // int64
    DTYPE_U8,       // uint8
} dtype_t;

// Tensor metadata
typedef struct {
    char* name;              // Tensor name (e.g., "model.layers.0.mlp.down_proj.weight")
    dtype_t dtype;           // Data type
    int64_t* shape;          // Dimensions [n_dims]
    int n_dims;              // Number of dimensions
    int64_t offset_start;    // Byte offset in file (after header)
    int64_t offset_end;      // End byte offset
    size_t num_elements;     // Total elements (product of shape)
} tensor_metadata_t;

// Safetensors file handle
typedef struct {
    int fd;                  // File descriptor
    uint8_t* data;           // Memory-mapped data
    size_t file_size;        // Total file size
    size_t data_offset;      // Offset where tensor data starts (8 + header_size)

    tensor_metadata_t* tensors;  // Array of tensor metadata
    int num_tensors;             // Number of tensors
} safetensors_t;

// Load safetensors file (memory-mapped, zero-copy)
safetensors_t* safetensors_load(const char* path);

// Find tensor by name
tensor_metadata_t* safetensors_find_tensor(safetensors_t* st, const char* name);

// Get tensor data as float32 (converts from BF16/F16 if needed)
float* safetensors_get_tensor_f32(safetensors_t* st, const char* name);

// Get tensor data as raw pointer (no conversion)
void* safetensors_get_tensor_raw(safetensors_t* st, const char* name);

// Free resources
void safetensors_free(safetensors_t* st);

// Print tensor info (for debugging)
void safetensors_print_info(safetensors_t* st);

// Save safetensors file
// Writes modified tensor data back to a new file
int safetensors_save(safetensors_t* st, const char* path);

#endif // SAFETENSORS_H
