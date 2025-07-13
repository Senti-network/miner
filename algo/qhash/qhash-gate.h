#ifndef QHASH_GATE_H__
#define QHASH_GATE_H__

#include "algo-gate-api.h"

#define NUM_QUBITS 16
#define NUM_LAYERS 2
#define SHA256_BLOCK_SIZE 32
#define INPUT_SIZE 80

bool register_qhash_algo(algo_gate_t *gate);

int qhash_hash(void *output, const void *input, int length);

// Simulator-specific functions

/**
 * Initializes all necessary simulation variables for each mining thread.
 * @param thr_id id of the thread being initialized
 * @return       true if the initialization was successful and false otherwise
 */
bool qhash_thread_init(int thr_id);

/**
 * Runs the quantum simulation and computes qubit expectations.
 * @param data         input hash for the circuit (each byte represents a single nibble for angle parametrization)
 * @param expectations output parameter for Z-basis expectations for each circuit qubit
 */
void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE], double expectations[NUM_QUBITS]);

#endif