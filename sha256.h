#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
	WORD h[8];
	BYTE b[32];
}SHA256;

//----------- FUNCTION DECLARATION --------
__host__ __device__ void sha256_transform(SHA256 *ctx, const BYTE *msg);
__host__ __device__ void sha256(SHA256 *ctx, const BYTE *msg, size_t len);
// initialize device-side k constant
void sha256_init_constants(void);

// midstate optimization helpers
void sha256_compute_midstate(const unsigned char *chunk64, WORD midstate[8]);
__host__ __device__ void sha256_transform_from_state(const WORD *state, const BYTE *chunk, SHA256 *out_ctx);
__host__ __device__ void sha256_finalize_from_midstate(SHA256 *ctx, const WORD midstate[8], const BYTE *last16bytes);

#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //__SHA256_HEADER__
