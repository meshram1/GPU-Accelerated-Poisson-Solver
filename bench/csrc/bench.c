// bench.c - time run_gpu vs cupoisson
// gcc -O2 bench.c -o bench

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static double run_one(const char *bin) {
    char cmd[512];
    sprintf(cmd, "%s > /dev/null 2>&1", bin);
    double t0 = now_ms();
    int rc = system(cmd);
    if (rc != 0) printf("warning: %s exited %d\n", bin, rc);
    return now_ms() - t0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: %s <ours_bin> <cupoisson_bin> [reps]\n", argv[0]);
        return 1;
    }
    int reps = (argc > 3) ? atoi(argv[3]) : 5;

    // discard one warm-up of each
    run_one(argv[1]);
    run_one(argv[2]);

    double so = 0.0, sc = 0.0;
    for (int i = 0; i < reps; i++) {
        double a = run_one(argv[1]);
        double b = run_one(argv[2]);
        printf("run %d: ours %.1f ms, cupoisson %.1f ms\n", i+1, a, b);
        so += a;
        sc += b;
    }

    double avg_o = so / reps;
    double avg_c = sc / reps;
    printf("\navg: ours %.1f ms, cupoisson %.1f ms\n", avg_o, avg_c);
    printf("speedup (ours / cupoisson): %.2f\n", avg_o / avg_c);
    return 0;
}
