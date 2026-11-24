/* CPU baseline for the Phase 0 comparison: time GMP's mpz_powm on a base-2 Fermat
 * exponentiation 2^(n-1) mod n for random odd n of a given size.
 * Build: gcc -O2 -o gmp_modexp gmp_modexp.c -lgmp
 * Usage: gmp_modexp [bits=2029] [count=100] [seed=1]
 * Prints one line: bits, count, ms per test, tests/s on this one thread.
 * Not the project's code and not constant-time: a native big-integer reference point only. */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    unsigned bits = argc > 1 ? (unsigned)atoi(argv[1]) : 2029;
    int count = argc > 2 ? atoi(argv[2]) : 100;
    unsigned long seed = argc > 3 ? strtoul(argv[3], 0, 10) : 1;
    gmp_randstate_t st;
    gmp_randinit_mt(st);
    gmp_randseed_ui(st, seed);
    mpz_t n, e, two, r, acc;
    mpz_inits(n, e, two, r, acc, NULL);
    mpz_set_ui(two, 2);
    int prp_like = 0; /* checksum so the compiler can't drop the work */
    double t0 = now();
    for (int i = 0; i < count; i++) {
        mpz_urandomb(n, st, bits - 1);
        mpz_setbit(n, bits - 1);
        mpz_setbit(n, 0);
        mpz_sub_ui(e, n, 1);
        mpz_powm(r, two, e, n);
        if (mpz_cmp_ui(r, 1) == 0) prp_like++;
    }
    double dt = now() - t0;
    printf("bits=%u count=%d ms/test=%.3f tests/s=%.1f (prp-like hits: %d)\n", bits, count, dt * 1e3 / count, count / dt, prp_like);
    return 0;
}
