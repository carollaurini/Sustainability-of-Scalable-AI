/**
 * Solves ∇u = f on [a, b] x [a, b] for a function u that satisfies
 * u(a, y) = u(b, y)
 * u(x, a) = u(x, b)
 *
 * for all x, y.
 **/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

double wtime(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1e-6 * (double)tv.tv_usec  + (double)tv.tv_sec;
}

int min(int a, int b)
{
    return (a < b) ? a : b;
}

double u(double x, double y)
{
    return sin(x + y);
}

double f(double x, double y)
{
    return -2.0 * sin(x + y);
}

void set_zero(size_t m, size_t n, double *x)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            x[i * n + j] = 0;
        }
    }
}

/* Turns 2D index (i, j) of an ... x n array with ghost-cells into
 * the offset from the start of the array. */
#define GHOST(i, j, n) ((i) + 1) * ((n) + 2) + ((j) + 1)

void update_ghost_cells(size_t m, size_t n, double *x)
{
    for (size_t k = 0; k < m; k++) {
        x[GHOST(k, -1, n)] = x[GHOST(k    , n - 1, n)];
        x[GHOST(k,  n, n)] = x[GHOST(k    , 0    , n)];
    }
    for (ssize_t k = -1; k < (long)(n + 1); k++) {
        x[GHOST(-1, k, n)] = x[GHOST(m - 1, k    , n)];
        x[GHOST(m , k, n)] = x[GHOST(0    , k    , n)];
    }
}

double *discretize_spooky(size_t m, size_t n, double h, double a,
                          double (*fun)(double, double))
{
    double *dis = malloc((m + 2) * (n + 2) * sizeof(double));

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double x = a + i * h;
            double y = a + j * h;
            dis[GHOST(i, j, n)] = fun(x, y);
        }
    }

    update_ghost_cells(m, n, dis);
    return dis;
}

/**
 * Loop body of sor function
 **/
#define SOR_BODY(n, u_dis, f_dis, i, j, omega, h)                              \
{                                                                              \
    u_dis[GHOST(i, j, n)] =                                                    \
       (1 - omega) *  u_dis[GHOST(i    , j    , n)] +                          \
       omega / 4   * (u_dis[GHOST(i - 1, j    , n)] +                          \
                      u_dis[GHOST(i + 1, j    , n)] +                          \
                      u_dis[GHOST(i    , j - 1, n)] +                          \
                      u_dis[GHOST(i    , j + 1, n)] -                          \
                      h * h * f_dis[GHOST(i, j, n)]);                          \
}

void sor(double *u_dis, double *f_dis, 
         size_t m, size_t n, double omega, double h)
{
    /**
     * Update black
     **/
    for (size_t i = 0; i < m - 1; i += 2) {
        /* Even rows */
        for (size_t j = 0; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, i, j, omega, h);
        }
        /* Odd rows */
        for (size_t j = 1; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, i + 1, j, omega, h);
        }
    }
    if (m % 2 == 1) {
        for (size_t j = 0; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, m - 1, j, omega, h);
        }
    }
    update_ghost_cells(m, n, u_dis);

    /**
     * Update red
     **/
    for (size_t i = 0; i < m - 1; i += 2) {
        /* Even rows */
        for (size_t j = 1; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, i, j, omega, h);
        }
        /* Odd rows */
        for (size_t j = 0; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, i + 1, j, omega, h);
        }
    }
    if (m % 2 == 1) {
        for (size_t j = 1; j < n; j += 2) {
            SOR_BODY(n, u_dis, f_dis, m - 1, j, omega, h);
        }
    }
    update_ghost_cells(m, n, u_dis);
}

int sor_solve(double *u_dis, double *f_dis, 
              size_t m, size_t n, double h, int max_iter)
{
    /**
     * Math black magic computing the number of iterations necessary for
     * convergence.
     **/
    double log_spectral_radius = log1p(-2.0 * sin(M_PI * h) /
                                    (1.0 + sin(M_PI * h)));
    int iter = 200.0 * log(h) / log_spectral_radius;
    iter = min(iter, max_iter);

    double omega = 2.0 / (1.0 + sin(M_PI * h));
    for (int t = 0; t < iter; t++) {
        sor(u_dis, f_dis, m, n, omega, h);
    }

    return iter;
}

double Linf(double *u_dis, size_t m, size_t n)
{
    double maximum = -1;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            maximum = fmax(fabs(u_dis[GHOST(i, j, n)]), maximum);
        }
    }
    return maximum;
}

int main(int argc, char **argv)
{
	clock_t begin = clock();
    if (argc != 3) {
        printf("Usage: %s <N> <MAX_ITER>\n"
               "\tN       : Discretize f on an N x N grid.\n"
              "\tMAX_ITER: Run for at most this many iterations\n",
               argv[0]);
        return EXIT_FAILURE;
    }

    size_t n     = atol(argv[1]);
    int max_iter = atoi(argv[2]);

    double a = 0;
    double b = 2 * M_PI;
    double h = (b - a) / n;

    double *f_dis = discretize_spooky(n, n, h, a, f);
    double *u_dis = malloc((n + 2) * (n + 2) * sizeof(double));

    set_zero(n + 2, n + 2, u_dis);

    double start = wtime();
    int iter = sor_solve(u_dis, f_dis, n, n, h, max_iter);
    double stop  = wtime();

    size_t size_of_array = n * n * sizeof(double);
    
    
    FILE *file;
    
  	char concat[100] = "./out_";
  	printf(argv[1]);
	char* c1 = argv[1];
	char c2[10] = ".txt"; 
	
  	strcat(concat, c1);
	strcat(concat, c2);	
	
    file = fopen(concat, "a+");
    
    
    //Bandwidth: GB/s
    fprintf(file, "%lf;",
            3.0 * size_of_array * iter / 1e9 / (stop - start));
            
    //Compute rate: GFLOPS
    fprintf(file, "%lf;",
            11.0 * n * n * iter / 1e9 / (stop - start));
    free(f_dis);

    double *real_answer = discretize_spooky(n, n, h, a, u);
    double norm_real = Linf(real_answer, n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            u_dis[GHOST(i, j, n)] -= real_answer[GHOST(i, j, n)];
        }
    }
    double norm_diff = Linf(u_dis, n, n);
    
    //Relative error
    fprintf(file, "%e;", 
            norm_diff / norm_real);	
	
    //printf("%lf\n", 3.0 * size_of_array * iter / 1e9 / (stop - start));

    free(u_dis);
    free(real_answer);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
	
	//Time
    fprintf(file, "%f\n", time_spent);
	fclose(file);
	
    return EXIT_SUCCESS;
    system("pause");
}
