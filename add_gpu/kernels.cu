__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}