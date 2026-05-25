// train_cuda.h - CUDA Training Pipeline Interface
// Part of TETYAH-PRIME's native training and inference engine

#ifndef TRAIN_CUDA_H
#define TRAIN_CUDA_H

#include "train.h"
#include "cuda_train_context.h"

// Initialize CUDA training context
// Returns NULL if CUDA is not available
cuda_train_context_t* cuda_train_init(int device_id);

// Free CUDA training context
void cuda_train_free(cuda_train_context_t* ctx);

// Synchronize CUDA stream
void cuda_train_sync(cuda_train_context_t* ctx);

// Main training step using CUDA
// Returns loss for this step
float cuda_train_step_gpu_full(train_state_t* state, int* tokens, int num_tokens,
                                cuda_train_context_t* ctx);

// Download weights from GPU to CPU (for checkpointing)
void cuda_download_full_weights(cuda_train_context_t* ctx, train_state_t* state);

// Download gradients from GPU to CPU (for gradient snapshot visualization)
void cuda_download_gpu_gradients(cuda_train_context_t* ctx, train_state_t* state);

// LoRA training step using CUDA
// Returns loss for this step
float cuda_train_step_gpu_lora(train_state_t* state, int* tokens, int num_tokens,
                                cuda_train_context_t* ctx);

// Packed batched training steps (batch >= 1)
// tokens_packed: contiguous token stream [seq0_tok0, seq0_tok1, ..., seq1_tok0, ...]
// total_tokens: sum of all sequence lengths
// seq_starts: [batch_size+1] prefix sum — seq_starts[i] = start offset of sequence i
// Returns average per-token loss across batch
float cuda_train_step_gpu_full_batch(train_state_t* state,
                                      int* tokens_packed, int total_tokens,
                                      int* seq_starts, int batch_size,
                                      cuda_train_context_t* ctx);

float cuda_train_step_gpu_lora_batch(train_state_t* state,
                                      int* tokens_packed, int total_tokens,
                                      int* seq_starts, int batch_size,
                                      cuda_train_context_t* ctx);

// Evaluate on a token sequence (forward + loss only, no backward/optimizer)
// Returns average cross-entropy loss
float cuda_eval_forward(train_state_t* state, int* tokens, int num_tokens,
                         cuda_train_context_t* ctx);

// Evaluate on full validation set, returns average loss and perplexity
// val_tokens/num_val_tokens: flat token array
// offsets/num_samples: sample boundaries (same format as train data)
// max_seq_len: truncation length
// max_eval_samples: 0 = all, >0 = cap (evenly spaced across val set)
float cuda_evaluate(train_state_t* state, int* val_tokens, int num_val_tokens,
                     int* offsets, int num_samples, int max_seq_len,
                     int max_eval_samples,
                     cuda_train_context_t* ctx, float* out_perplexity);

#endif // TRAIN_CUDA_H
