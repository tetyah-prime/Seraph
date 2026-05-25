// safetensors_cjson.c - Using cJSON library for robust parsing
#include "../include/safetensors.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// BFloat16 to Float32 conversion
static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

// Float16 to Float32 conversion
static float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t frac = f16 & 0x3FF;

    uint32_t f32_bits;
    if (exp == 0) {
        if (frac == 0) {
            f32_bits = sign << 31;
        } else {
            exp = 1;
            while (!(frac & 0x400)) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
            f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f32_bits = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
        f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }

    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

static dtype_t parse_dtype(const char* dtype_str) {
    if (strcmp(dtype_str, "F32") == 0) return DTYPE_F32;
    if (strcmp(dtype_str, "F16") == 0) return DTYPE_F16;
    if (strcmp(dtype_str, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(dtype_str, "I8") == 0) return DTYPE_I8;
    if (strcmp(dtype_str, "I16") == 0) return DTYPE_I16;
    if (strcmp(dtype_str, "I32") == 0) return DTYPE_I32;
    if (strcmp(dtype_str, "I64") == 0) return DTYPE_I64;
    if (strcmp(dtype_str, "U8") == 0) return DTYPE_U8;
    return DTYPE_UNKNOWN;
}

safetensors_t* safetensors_load(const char* path) {
    // Open file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return NULL;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return NULL;
    }
    size_t file_size = st.st_size;

    // Load into ram instead of mmap
    printf("  Loading %zu bytes into RAM...\n", file_size);
    uint8_t* data = malloc(file_size);
    if (!data) {
        perror("malloc");
        close(fd);
        return NULL;
    }

    // Read entire file into RAM
    size_t bytes_read = 0;
    while (bytes_read < file_size) {
        ssize_t n = read(fd, data + bytes_read, file_size - bytes_read);
        if (n <= 0) {
            perror("read");
            free(data);
            close(fd);
            return NULL;
        }
        bytes_read += n;
    }
    printf("  ✅ Loaded into RAM\n");

    // Read 8-byte header size
    if (file_size < 8) {
        fprintf(stderr, "File too small\n");
        free(data);
        close(fd);
        return NULL;
    }

    uint64_t header_size;
    memcpy(&header_size, data, 8);

    if (8 + header_size > file_size) {
        fprintf(stderr, "Invalid header size\n");
        free(data);
        close(fd);
        return NULL;
    }

    // Extract JSON header
    char* header_json = malloc(header_size + 1);
    memcpy(header_json, data + 8, header_size);
    header_json[header_size] = '\0';

    // Parse JSON with cJSON
    cJSON* root = cJSON_Parse(header_json);
    free(header_json);

    if (!root) {
        fprintf(stderr, "Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        free(data);
        close(fd);
        return NULL;
    }

    // Count tensors (exclude __metadata__)
    int num_tensors = 0;
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") != 0) {
            num_tensors++;
        }
    }

    // Allocate tensor array
    tensor_metadata_t* tensors = calloc(num_tensors, sizeof(tensor_metadata_t));

    // Parse tensors
    int tensor_idx = 0;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") == 0) continue;

        tensor_metadata_t* tensor = &tensors[tensor_idx++];
        tensor->name = strdup(item->string);

        // Get dtype
        cJSON* dtype = cJSON_GetObjectItem(item, "dtype");
        if (dtype && cJSON_IsString(dtype)) {
            tensor->dtype = parse_dtype(dtype->valuestring);
        }

        // Get shape
        cJSON* shape = cJSON_GetObjectItem(item, "shape");
        if (shape && cJSON_IsArray(shape)) {
            tensor->n_dims = cJSON_GetArraySize(shape);
            tensor->shape = malloc(tensor->n_dims * sizeof(int64_t));
            tensor->num_elements = 1;

            for (int i = 0; i < tensor->n_dims; i++) {
                cJSON* dim = cJSON_GetArrayItem(shape, i);
                tensor->shape[i] = dim->valueint;
                tensor->num_elements *= tensor->shape[i];
            }
        }

        // Get data_offsets (use valuedouble for int64 values > INT32_MAX)
        cJSON* offsets = cJSON_GetObjectItem(item, "data_offsets");
        if (offsets && cJSON_IsArray(offsets) && cJSON_GetArraySize(offsets) == 2) {
            tensor->offset_start = (int64_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
            tensor->offset_end = (int64_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
        }
    }

    cJSON_Delete(root);

    // Create safetensors handle
    safetensors_t* sft = malloc(sizeof(safetensors_t));
    sft->fd = fd;
    sft->data = data;
    sft->file_size = file_size;
    sft->data_offset = 8 + header_size;
    sft->tensors = tensors;
    sft->num_tensors = num_tensors;

    return sft;
}

tensor_metadata_t* safetensors_find_tensor(safetensors_t* st, const char* name) {
    for (int i = 0; i < st->num_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            return &st->tensors[i];
        }
    }
    return NULL;
}

void* safetensors_get_tensor_raw(safetensors_t* st, const char* name) {
    tensor_metadata_t* tensor = safetensors_find_tensor(st, name);
    if (!tensor) return NULL;

    return st->data + st->data_offset + tensor->offset_start;
}

float* safetensors_get_tensor_f32(safetensors_t* st, const char* name) {
    tensor_metadata_t* tensor = safetensors_find_tensor(st, name);
    if (!tensor) return NULL;

    uint8_t* raw_data = st->data + st->data_offset + tensor->offset_start;
    float* f32_data = malloc(tensor->num_elements * sizeof(float));

    if (tensor->dtype == DTYPE_F32) {
        memcpy(f32_data, raw_data, tensor->num_elements * sizeof(float));
    } else if (tensor->dtype == DTYPE_BF16) {
        uint16_t* bf16_data = (uint16_t*)raw_data;
        for (size_t i = 0; i < tensor->num_elements; i++) {
            f32_data[i] = bf16_to_f32(bf16_data[i]);
        }
    } else if (tensor->dtype == DTYPE_F16) {
        uint16_t* f16_data = (uint16_t*)raw_data;
        for (size_t i = 0; i < tensor->num_elements; i++) {
            f32_data[i] = f16_to_f32(f16_data[i]);
        }
    } else {
        fprintf(stderr, "Unsupported dtype for F32 conversion\n");
        free(f32_data);
        return NULL;
    }

    return f32_data;
}

void safetensors_free(safetensors_t* st) {
    if (!st) return;

    for (int i = 0; i < st->num_tensors; i++) {
        free(st->tensors[i].name);
        free(st->tensors[i].shape);
    }
    free(st->tensors);

    free(st->data);  
    close(st->fd);
    free(st);
}

void safetensors_print_info(safetensors_t* st) {
    printf("Safetensors file info:\n");
    printf("  File size: %zu MB\n", st->file_size / 1024 / 1024);
    printf("  Data offset: %zu bytes\n", st->data_offset);
    printf("  Number of tensors: %d\n\n", st->num_tensors);

    for (int i = 0; i < st->num_tensors && i < 20; i++) {
        tensor_metadata_t* t = &st->tensors[i];
        printf("[%d] %s\n", i, t->name);
        printf("    dtype: %d, shape: [", t->dtype);
        for (int j = 0; j < t->n_dims; j++) {
            printf("%ld", t->shape[j]);
            if (j < t->n_dims - 1) printf(", ");
        }
        printf("], elements: %zu\n", t->num_elements);
        printf("    offsets: [%ld, %ld], size: %zu bytes\n",
               t->offset_start, t->offset_end,
               (size_t)(t->offset_end - t->offset_start));
    }

    if (st->num_tensors > 20) {
        printf("  ... and %d more tensors\n", st->num_tensors - 20);
    }
}

static const char* dtype_to_string(dtype_t dtype) {
    switch (dtype) {
        case DTYPE_F32: return "F32";
        case DTYPE_F16: return "F16";
        case DTYPE_BF16: return "BF16";
        case DTYPE_I8: return "I8";
        case DTYPE_I16: return "I16";
        case DTYPE_I32: return "I32";
        case DTYPE_I64: return "I64";
        case DTYPE_U8: return "U8";
        default: return "UNKNOWN";
    }
}

int safetensors_save(safetensors_t* st, const char* path) {
    // Build JSON header from tensor metadata
    cJSON* root = cJSON_CreateObject();

    for (int i = 0; i < st->num_tensors; i++) {
        tensor_metadata_t* t = &st->tensors[i];

        cJSON* tensor_obj = cJSON_CreateObject();
        cJSON_AddStringToObject(tensor_obj, "dtype", dtype_to_string(t->dtype));

        cJSON* shape = cJSON_CreateArray();
        for (int j = 0; j < t->n_dims; j++) {
            cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)t->shape[j]));
        }
        cJSON_AddItemToObject(tensor_obj, "shape", shape);

        cJSON* offsets = cJSON_CreateArray();
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)t->offset_start));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)t->offset_end));
        cJSON_AddItemToObject(tensor_obj, "data_offsets", offsets);

        cJSON_AddItemToObject(root, t->name, tensor_obj);
    }

    // Generate JSON string
    char* header_json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!header_json) {
        fprintf(stderr, "ERROR: Failed to generate JSON header\n");
        return -1;
    }

    uint64_t header_size = strlen(header_json);
    size_t data_size = st->file_size - st->data_offset;

    printf("  Writing %zu MB to %s...\n", (8 + header_size + data_size) / 1024 / 1024, path);

    // Open output file
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("fopen");
        free(header_json);
        return -1;
    }

    // Write header size (8 bytes, little endian)
    fwrite(&header_size, sizeof(uint64_t), 1, f);

    // Write JSON header
    fwrite(header_json, 1, header_size, f);
    free(header_json);

    // Write tensor data (from RAM buffer)
    fwrite(st->data + st->data_offset, 1, data_size, f);

    fclose(f);
    printf("  ✅ Saved successfully\n");

    return 0;
}
