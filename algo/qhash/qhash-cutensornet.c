#include <complex.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cutensornet.h>
#include <math.h>
#include <stdbool.h>

#include "qhash-gate.h"

#define HANDLE_CUTN_ERROR(x)                                                     \
    {                                                                            \
        cutensornetStatus_t err = x;                                             \
        if (err != CUTENSORNET_STATUS_SUCCESS)                                   \
            fprintf(stdout, "cuTensorNet Error: %s in line %d\n",                \
                    cutensornetGetErrorString(err), __LINE__);                   \
    }
#define HANDLE_CUDA_ERROR(x)                                                     \
    {                                                                            \
        cudaError_t err = x;                                                     \
        if (err != cudaSuccess)                                                  \
            fprintf(stdout, "CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    }

static const cuComplex matrixX[] = {{0.0f,0.0f},{1.0f,0.0f},{1.0f,0.0f},{0.0f,0.0f}};
static const cuComplex matrixZ[] = {{1.0f,0.0f},{0.0f,0.0f},{0.0f,0.0f},{-1.0f,0.0f}};

static __thread cutensornetHandle_t handle;
static __thread cutensornetState_t state;
static __thread void* scratch = NULL;
static __thread size_t scratchSize = 0;

bool qhash_cutensornet_thread_init(int thr_id)
{
    HANDLE_CUTN_ERROR(cutensornetCreate(&handle));
    const int32_t dims[NUM_QUBITS] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    HANDLE_CUTN_ERROR(cutensornetCreateState(handle, CUTENSORNET_STATE_PURITY_PURE,
                                             NUM_QUBITS, dims, CUDA_C_32F, &state));
    size_t freeSize,totalSize; 
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize,&totalSize));
    scratchSize = (freeSize/2) & ~255ULL;
    HANDLE_CUDA_ERROR(cudaMalloc(&scratch,scratchSize));
    return true;
}

static void apply_gate_1q(int qubit, const cuComplex gate[4])
{
    int32_t modes[] = {qubit};
    int64_t id;
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(handle, state, 1, modes,
                                                          gate, NULL, 1, 0, 1, &id));
}

static void apply_gate_cnot(int control, int target)
{
    int32_t modes[] = {control,target};
    cuComplex cx[16] = {};
    cx[0] = (cuComplex){1.0f,0.0f};
    cx[5] = (cuComplex){1.0f,0.0f};
    cx[10]= (cuComplex){1.0f,0.0f};
    cx[15]= (cuComplex){1.0f,0.0f};
    cx[15-5] = (cuComplex){1.0f,0.0f}; /* not used but keep */
    int64_t id;
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(handle, state, 2, modes,
                                                          cx, NULL, 1, 0, 1, &id));
}

static void main_circuit(const unsigned char data[2 * SHA256_BLOCK_SIZE])
{
    for (size_t l = 0; l < NUM_LAYERS; ++l) {
        for (size_t i = 0; i < NUM_QUBITS; ++i) {
            float ry = -data[(2*l*NUM_QUBITS+i) % (2*SHA256_BLOCK_SIZE)] * M_PI / 16.0f;
            float rz = -data[((2*l+1)*NUM_QUBITS+i) % (2*SHA256_BLOCK_SIZE)] * M_PI / 16.0f;
            cuComplex gateRY[4] = {{cosf(ry/2),0.0f},{-sinf(ry/2),0.0f},{sinf(ry/2),0.0f},{cosf(ry/2),0.0f}};
            cuComplex gateRZ[4] = {{cosf(rz/2),-sinf(rz/2)},{0,0},{0,0},{cosf(-rz/2),sinf(-rz/2)}};
            apply_gate_1q(i, gateRY);
            apply_gate_1q(i, gateRZ);
        }
        for (size_t i = 0; i < NUM_QUBITS - 1; ++i) {
            apply_gate_cnot(i, i+1);
        }
    }
}

static void get_expectations(double expectations[NUM_QUBITS])
{
    for (size_t i = 0; i < NUM_QUBITS; ++i) {
        cutensornetNetworkOperator_t op;
        int32_t dims[NUM_QUBITS];
        for (int k=0;k<NUM_QUBITS;k++) dims[k]=2;
        HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(handle, NUM_QUBITS,dims, CUDA_C_32F, &op));
        int32_t numModes = 1;
        int32_t mode = i;
        const void* gateData = matrixZ;
        cuComplex coeff = {1.0f,0.0f};
        int64_t id;
        HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(handle, op, coeff,
                              numModes, &numModes, &mode, NULL, &gateData, &id));
        cutensornetStateExpectation_t exp;
        HANDLE_CUTN_ERROR(cutensornetCreateExpectation(handle, state, op, &exp));
        cutensornetWorkspaceDescriptor_t w;
        HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(handle,&w));
        HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(handle, exp, scratchSize, w, 0));
        HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, w, CUTENSORNET_MEMSPACE_DEVICE,
                                                         CUTENSORNET_WORKSPACE_SCRATCH, scratch, scratchSize));
        cuDoubleComplex val={0,0}, norm={0,0};
        HANDLE_CUTN_ERROR(cutensornetExpectationCompute(handle, exp, w, &val,&norm,0));
        expectations[i] = (norm.x!=0)? val.x/norm.x : 0.0;
        cutensornetDestroyWorkspaceDescriptor(w);
        cutensornetDestroyExpectation(exp);
        cutensornetDestroyNetworkOperator(op);
    }
}

void run_simulation_cutensornet(const unsigned char data[2 * SHA256_BLOCK_SIZE],
                                double expectations[NUM_QUBITS])
{
    HANDLE_CUTN_ERROR(cutensornetStateInitialize(handle, state,
                                CUTENSORNET_STATE_VECTOR_TYPE_ZERO, NULL));
    main_circuit(data);
    get_expectations(expectations);
}
