#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#if USE_THREADS
# include <pthread.h>
#endif

#include "freesasa_internal.h"
#include "nb.h"

const double TWOPI = 2*M_PI;

//calculation parameters and data (results stored in *sasa)
typedef struct {
    int n_atoms;
    double *radii; //including probe
    const coord_t *xyz;
    nb_list *adj;
    int n_slices_per_atom;
    double *sasa; // results
} lr_data;

typedef struct {
    int first_atom;
    int last_atom;
    lr_data *lr;
} lr_thread_interval;

#if USE_THREADS
static int lr_do_threads(int n_threads, lr_data*);
static void *lr_thread(void *arg);
#endif

/** Returns the are of atom i */
static double
atom_area(lr_data *lr,int i);

/** Sum of exposed arcs based on buried arc intervals arc, assumes no
    intervals cross zero */
static double
exposed_arc_length(double *restrict arc, int n);

/** Release contenst of lr_data pointer*/
static void
release_lr(lr_data *lr)
{
    free(lr->radii);
    freesasa_nb_free(lr->adj);
    lr->radii = NULL;
    lr->adj = NULL;
}

/** Initialize object to be used for L&R calculation */
static int
init_lr(lr_data *lr,
        double *sasa,
        const coord_t *xyz,
        const double *atom_radii,
        double probe_radius,
        int n_slices_per_atom)
{
    const int n_atoms = freesasa_coord_n(xyz);

    lr->n_atoms = n_atoms;
    lr->xyz = xyz;
    lr->adj = NULL;
    lr->n_slices_per_atom = n_slices_per_atom;
    lr->sasa = sasa;

    lr->radii = malloc(sizeof(double)*n_atoms);
    if (lr->radii == NULL) {
        return mem_fail();
    }

    //init some arrays
    for (int i = 0; i < n_atoms; ++i) {
        lr->radii[i] = atom_radii[i] + probe_radius;
        sasa[i] = 0.;
    }

    // determine which atoms are neighbours
    lr->adj = freesasa_nb_new(xyz, lr->radii);

    if (lr->adj == NULL) {
        release_lr(lr);
        return FREESASA_FAIL;
    }

    return FREESASA_SUCCESS;

}

int
freesasa_lee_richards(double *sasa,
                      const coord_t *xyz,
                      const double *atom_radii,
                      const freesasa_parameters *param)
{
    assert(sasa);
    assert(xyz);
    assert(atom_radii);

    if (param == NULL) param = &freesasa_default_parameters;

    int return_value = FREESASA_SUCCESS,
        n_atoms = freesasa_coord_n(xyz),
        n_threads = param->n_threads,
        resolution = param->lee_richards_n_slices;
    double probe_radius = param->probe_radius;
    lr_data lr;

    if (resolution <= 0)
        return fail_msg("%f slices per atom invalid resolution in L&R, must be > 0\n", resolution);

    if (n_atoms == 0) {
        return freesasa_warn("in %s(): empty coordinates", __func__);
    }
    if (n_threads > n_atoms) {
        n_threads = n_atoms;
        freesasa_warn("no sense in having more threads than atoms, only using %d threads",
                      n_threads);
    }
    
    if(init_lr(&lr, sasa, xyz, atom_radii, probe_radius, resolution))
        return FREESASA_FAIL;
    
    if (n_threads > 1) {
#if USE_THREADS
        return_value = lr_do_threads(n_threads, &lr);
#else
        return_value = freesasa_warn("in %s(): program compiled for single-threaded use, "
                                     "but multiple threads were requested, will "
                                     "proceed in single-threaded mode\n",
                                     __func__);
        n_threads = 1;
#endif /* pthread */
    }
    if (n_threads == 1) {
        for (int i = 0; i < lr.n_atoms; ++i) {
            lr.sasa[i] = atom_area(&lr, i);
        }        
    }
    release_lr(&lr);
    return return_value;
}

#if USE_THREADS
static int
lr_do_threads(int n_threads,
              lr_data *lr)
{
    pthread_t thread[n_threads];
    lr_thread_interval t_data[n_threads];
    int n_perthread = lr->n_atoms/n_threads, res;
    int threads_created = 0, return_value = FREESASA_SUCCESS;
 
    for (int t = 0; t < n_threads; ++t) {
        t_data[t].first_atom = t*n_perthread;
        if (t == n_threads-1) {
            t_data[t].last_atom = lr->n_atoms - 1;
        } else {
            t_data[t].last_atom = (t+1)*n_perthread - 1;
        }
        t_data[t].lr = lr;
        res = pthread_create(&thread[t], NULL, lr_thread,
                             (void *) &t_data[t]);
        if (res) {
            return_value = fail_msg(freesasa_thread_error(res));
            break;
        }
        ++threads_created;
    }
    for (int t = 0; t < threads_created; ++t) {
        res = pthread_join(thread[t],NULL);
        if (res) {
            return_value = fail_msg(freesasa_thread_error(res));
        }
    }
    return return_value;
}

static void*
lr_thread(void *arg)
{
    lr_thread_interval *ti = ((lr_thread_interval*) arg);
    for (int i = ti->first_atom; i <= ti->last_atom; ++i) {
        /* the different threads write to different parts of the
           array, so locking shouldn't be necessary */
        ti->lr->sasa[i] = atom_area(ti->lr, i);
    }
    pthread_exit(NULL);
}
#endif /* USE_THREADS */

static double
atom_area(lr_data *lr,
          int i)
{
    /* This function is large because a large number of pre-calculated
       arrays need to be accessed efficiently. Partially dereferenced
       here to make access more efficient.
    
       Variables are named according to the documentation (see page
       "Geometry of Lee & Richards' algorithm") */

    const int nni = lr->adj->nn[i];
    const double * restrict const v = freesasa_coord_all(lr->xyz);
    const double * restrict const R = lr->radii;
    const int * restrict const nbi = lr->adj->nb[i];
    const double * restrict const xydi = lr->adj->xyd[i];
    const double * restrict const xdi = lr->adj->xd[i];
    const double * restrict const ydi = lr->adj->yd[i];
    const double zi = v[3*i+2], Ri = R[i];
    const int ns = lr->n_slices_per_atom;
    double arc[nni*4], z_nb[nni], R_nb[nni];
    double z, delta, sasa = 0;
    
    for (int j = 0; j < nni; ++j) {
        z_nb[j] = v[3*nbi[j]+2];
        R_nb[j] = R[nbi[j]];
    }
    
    delta = 2*Ri/ns;
    z = zi-Ri-0.5*delta;
    for (int islice = 0; islice < ns; ++islice) {
        z += delta;
        const double di = fabs(zi - z);
        const double Ri_prime2 = Ri*Ri-di*di;
        if (Ri_prime2 < 0 ) continue; // handle round-off errors
        const double Ri_prime = sqrt(Ri_prime2);
        if (Ri_prime <= 0) continue; // more round-off errors
        int n_arcs = 0, is_buried = 0;
        for (int j = 0; j < nni; ++j) {
            const double zj = z_nb[j];
            const double dj = fabs(zj - z);
            const double Rj = R_nb[j];
            if (dj < Rj) {
                const double Rj_prime2 = Rj*Rj-dj*dj;
                const double Rj_prime = sqrt(Rj_prime2);
                const double dij = xydi[j];
                double alpha, beta, inf, sup;
                int narc2;
                if (dij >= Ri_prime + Rj_prime) { // atoms aren't in contact
                    continue;
                }
                if (dij + Ri_prime < Rj_prime) { // circle i is completely inside j
                    is_buried = 1;
                    break;
                }
                if (dij + Rj_prime < Ri_prime) { // circle j is completely inside i
                    continue;
                }
                // arc of circle i intersected by circle j
                alpha = acos ((Ri_prime2 + dij*dij - Rj_prime2)/(2.0*Ri_prime*dij));
                // position of mid-point of intersection along circle i
                beta = atan2 (ydi[j],xdi[j]) + M_PI;
                inf = beta - alpha;
                sup = beta + alpha;
                if (inf < 0) inf += TWOPI;
                if (sup > 2*M_PI) sup -= TWOPI;
                narc2 = 2*n_arcs;
                // store the arc, if arc passes 2*PI split into two
                if (sup < inf) {
                    //store arcs as contiguous pairs of angles
                    arc[narc2]   = 0;
                    arc[narc2+1] = sup;
                    //second arc
                    arc[narc2+2] = inf;
                    arc[narc2+3] = TWOPI;
                    n_arcs += 2;
                } else { 
                    arc[narc2]   = inf;
                    arc[narc2+1] = sup;
                    ++n_arcs;
                }
            }
        }
        if (is_buried == 0) {
            sasa += delta*Ri*exposed_arc_length(arc,n_arcs);
        }
    }
    return sasa;
}

//insertion sort (faster than qsort for these short lists)
inline static void
sort_arcs(double * restrict arc,
          int n) 
{
    double tmp[2];
    double *end = arc+2*n, *arcj, *arci;
    for (arci = arc+2; arci < end; arci += 2) {
        //memcpy(tmp,arci,2*sizeof(double));
        // this is much faster than memcpy for this small chunk
        *tmp = *arci;
        *(tmp+1) = *(arci+1);
        arcj = arci;
        while (arcj > arc && *(arcj-2) > tmp[0]) {
            //memcpy(arcj,arcj-2,2*sizeof(double)); 
            *arcj = *(arcj-2);
            *(arcj+1) = *(arcj-1);
            arcj -= 2;
        }
        //memcpy(arcj,tmp,2*sizeof(double));
        *arcj = *tmp;
        *(arcj+1) = *(tmp+1);
    }
}

// sort arcs by start-point, loop through them to sum parts of circle
// not covered by any of the arcs
inline static double
exposed_arc_length(double * restrict arc,
                   int n)
{
    if (n == 0) return TWOPI;
    double sum, sup, tmp;
    sort_arcs(arc,n);
    sum = arc[0];
    sup = arc[1];
    // in the following it is assumed that the arc[i2] <= arc[i2+1]
    for (int i2 = 2; i2 < 2*n; i2 += 2) {
        if (sup < arc[i2]) sum += arc[i2] - sup;
        tmp = arc[i2+1];
        if (tmp > sup) sup = tmp;
    } 
    return sum + TWOPI - sup;
}

#if USE_CHECK
#include <check.h>

static int
is_identical(const double *l1, const double *l2, int n) {
    for (int i = 0; i < n; ++i) {
        if (l1[i] != l2[i]) return 0;
    }
    return 1;
}

static int
is_sorted(const double *list,int n)
{
    for (int i = 0; i < n - 1; ++i) if (list[2*i] > list[2*i+1]) return 0;
    return 1;
}

START_TEST (test_sort_arcs) {
    double a_ref[] = {0,1,2,3}, b_ref[] = {-2,0,-1,0,-1,1};
    double a1[4] = {0,1,2,3}, a2[4] = {2,3,0,1};
    double b1[6] = {-2,0,-1,0,-1,1}, b2[6] = {-1,1,-2,0,-1,1};
    sort_arcs(a1,2);
    sort_arcs(a2,2);
    sort_arcs(b1,3);
    sort_arcs(b2,3);
    ck_assert(is_sorted(a1,2));
    ck_assert(is_sorted(a2,2));
    ck_assert(is_sorted(b1,3));
    ck_assert(is_sorted(b2,3));
    ck_assert(is_identical(a_ref,a1,4));
    ck_assert(is_identical(a_ref,a2,4));
    ck_assert(is_identical(b_ref,b1,6));
}
END_TEST

START_TEST (test_exposed_arc_length) 
{
    double a1[4] = {0,0.1*TWOPI,0.9*TWOPI,TWOPI}, a2[4] = {0.9*TWOPI,TWOPI,0,0.1*TWOPI};
    double a3[4] = {0,TWOPI,1,2}, a4[4] = {1,2,0,TWOPI};
    double a5[4] = {0.1*TWOPI,0.2*TWOPI,0.5*TWOPI,0.6*TWOPI};
    double a6[4] = {0.1*TWOPI,0.2*TWOPI,0.5*TWOPI,0.6*TWOPI};
    double a7[4] = {0.1*TWOPI,0.3*TWOPI,0.15*TWOPI,0.2*TWOPI};
    double a8[4] = {0.15*TWOPI,0.2*TWOPI,0.1*TWOPI,0.3*TWOPI};
    double a9[10] = {0.05,0.1, 0.5,0.6, 0,0.15, 0.7,0.8, 0.75,TWOPI};
    ck_assert(fabs(exposed_arc_length(a1,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a2,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a3,2)) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a4,2)) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a5,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a6,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a7,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a8,2) - 0.8*TWOPI) < 1e-10);
    ck_assert(fabs(exposed_arc_length(a9,5) - 0.45) < 1e-10);
    // can't think of anything more qualitatively different here
}
END_TEST

TCase *
test_LR_static()
{
    TCase *tc = tcase_create("sasa_lr.c static");
    tcase_add_test(tc, test_sort_arcs);
    tcase_add_test(tc, test_exposed_arc_length);

    return tc;
}

#endif // USE_CHECK
