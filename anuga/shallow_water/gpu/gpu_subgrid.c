// Subgrid terrain on the device (#508 prototype, mode 2, Euler step).
//
// gpu_subgrid_init maps the subface beds and the storage tables and allocates
// the draining limiter's scratch. gpu_evolve_one_euler_step then calls the
// subgrid edge flux and the draining limiter in place of the central flux, and
// the storage operator calls gpu_subgrid_storage_update after the forcing
// operators, so a step makes no host round trip. The kernels themselves are in
// core_kernels.c, shared with mode 1.

#include <stdlib.h>
#include <omp.h>
#include "gpu_domain.h"
#include "core_kernels.h"

int gpu_subgrid_init(struct gpu_domain *GD, anuga_int nsub, anuga_int m,
                     double *sub_z, double *sub_invn,
                     double *tz, double *tV, double *tS, double *tinvn,
                     double *n_base, double *cell_neff, int drain_limiter,
                     double cfl, double cap_exponent, double cap_min_depth,
                     double cap_min_fraction, double cap, double fr_cap) {
    if (!GD->gpu_initialized) {
        gpu_set_error(GD, "gpu_subgrid_init: the domain arrays are not mapped yet");
        return -1;
    }
    if (GD->subgrid != NULL) gpu_subgrid_finalize(GD);
    struct gpu_subgrid *S = (struct gpu_subgrid *) calloc(1, sizeof(struct gpu_subgrid));
    if (S == NULL) {
        gpu_set_error(GD, "gpu_subgrid_init: out of memory");
        return -1;
    }
    anuga_int n = GD->D.number_of_elements;
    S->nsub = nsub;
    S->m = m;
    S->sub_z = sub_z;
    S->sub_invn = sub_invn;
    S->tz = tz;
    S->tV = tV;
    S->tS = tS;
    S->tinvn = tinvn;
    S->n_base = n_base;
    S->cell_neff = (sub_invn != NULL) ? cell_neff : NULL;
    S->cfl = cfl;
    S->cap_exponent = cap_exponent;
    S->cap_min_depth = cap_min_depth;
    S->cap_min_fraction = cap_min_fraction;
    S->cap = cap;
    S->fr_cap = fr_cap;
    if (drain_limiter) {
        S->slot_flux = (double *) calloc((size_t) (9 * n), sizeof(double));
        S->cell_pres = (double *) calloc((size_t) (2 * n), sizeof(double));
        S->phi = (double *) calloc((size_t) n, sizeof(double));
        if (S->slot_flux == NULL || S->cell_pres == NULL || S->phi == NULL) {
            free(S->slot_flux); free(S->cell_pres); free(S->phi); free(S);
            gpu_set_error(GD, "gpu_subgrid_init: out of memory for the limiter scratch");
            return -1;
        }
    }
    // The friction the storage operator set on the host, and the bed and level
    // of set_initial_level, may be newer than the device copies.
    double *friction_cv = GD->D.friction_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *stage_cv = GD->D.stage_centroid_values;
    #pragma omp target update to(friction_cv[0:n], bed_cv[0:n], stage_cv[0:n])

    #pragma omp target enter data map(to: sub_z[0:3*n*nsub], tz[0:n*m], tV[0:n*m], tS[0:n*m], tinvn[0:n*m], n_base[0:n])
    if (S->sub_invn != NULL) {
        double *si = S->sub_invn, *cn = S->cell_neff;
        #pragma omp target enter data map(to: si[0:3*n*nsub], cn[0:n])
    }
    if (S->slot_flux != NULL) {
        double *sf = S->slot_flux, *cp = S->cell_pres, *ph = S->phi;
        #pragma omp target enter data map(to: sf[0:9*n], cp[0:2*n], ph[0:n])
    }
    GD->subgrid = S;
    return 0;
}

double gpu_subgrid_storage_update(struct gpu_domain *GD, double max_step) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL) return max_step;
    S->cap = core_subgrid_storage_update(&GD->D, S->m, S->tz, S->tV, S->tS, S->tinvn,
                                         S->n_base, S->cell_neff, S->cfl, S->cap_exponent,
                                         S->cap_min_depth, S->cap_min_fraction, max_step,
                                         S->tL, S->tlev, S->tR, S->tensor_form, S->fr_cap);
    return S->cap;
}

// Conveyance tensor tables (SubgridTables.add_tensor): from now on the storage
// update takes the friction from the tensor's conveyance along the discharge.
int gpu_subgrid_set_tensor(struct gpu_domain *GD, anuga_int tL, double *tlev, double *tR, int tensor_form) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL || S->tlev != NULL) {
        gpu_set_error(GD, "gpu_subgrid_set_tensor: needs gpu_subgrid_init first, once");
        return -1;
    }
    anuga_int n = GD->D.number_of_elements;
    #pragma omp target enter data map(to: tlev[0:n*tL], tR[0:n*tL*3])
    S->tL = tL;
    S->tlev = tlev;
    S->tR = tR;
    S->tensor_form = tensor_form;
    return 0;
}

// The pond store: tables and the volumes held go to the device, which owns the
// volumes from then on (gpu_subgrid_sync_ponds brings them back to the host).
int gpu_subgrid_set_ponds(struct gpu_domain *GD, const struct subgrid_ponds *ponds) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL || S->ponds != NULL) {
        gpu_set_error(GD, "gpu_subgrid_set_ponds: needs gpu_subgrid_init first, once");
        return -1;
    }
    struct subgrid_ponds *P = (struct subgrid_ponds *) malloc(sizeof(struct subgrid_ponds));
    if (P == NULL) {
        gpu_set_error(GD, "gpu_subgrid_set_ponds: out of memory");
        return -1;
    }
    *P = *ponds;
    anuga_int npd = P->npd, nc = P->nc, ntab = P->ntab;
    anuga_int *cells = P->cells, *cstart = P->cstart, *tstart = P->tstart;
    double *tb = P->tb, *tslope = P->tslope, *tvol = P->tvol, *tat = P->tat;
    double *zs = P->zs, *Vzs = P->Vzs, *vmax = P->vmax, *shigh = P->shigh, *zlow = P->zlow, *w = P->w;
    double *V = P->V, *dV = P->dV;
    #pragma omp target enter data map(to: cells[0:nc], cstart[0:nc+1], tstart[0:npd+1])
    #pragma omp target enter data map(to: tb[0:ntab], tslope[0:ntab], tvol[0:ntab], tat[0:ntab])
    #pragma omp target enter data map(to: zs[0:npd], Vzs[0:npd], vmax[0:npd], shigh[0:npd], zlow[0:npd], w[0:npd])
    #pragma omp target enter data map(to: V[0:npd], dV[0:npd])
    S->ponds = P;
    return 0;
}

double gpu_subgrid_pond_exchange(struct gpu_domain *GD, double dt) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL || S->ponds == NULL) return 0.0;
    return core_subgrid_pond_exchange(&GD->D, S->m, S->tz, S->tV, S->tS, S->ponds, dt);
}

void gpu_subgrid_sync_ponds(struct gpu_domain *GD) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL || S->ponds == NULL) return;
    anuga_int npd = S->ponds->npd;
    double *V = S->ponds->V;
    #pragma omp target update from(V[0:npd])
}

void gpu_subgrid_finalize(struct gpu_domain *GD) {
    struct gpu_subgrid *S = GD->subgrid;
    if (S == NULL) return;
    anuga_int n = GD->D.number_of_elements;
    anuga_int nsub = S->nsub, m = S->m;
    double *sub_z = S->sub_z, *tz = S->tz, *tV = S->tV, *tS = S->tS, *tinvn = S->tinvn, *n_base = S->n_base;
    #pragma omp target exit data map(delete: sub_z[0:3*n*nsub], tz[0:n*m], tV[0:n*m], tS[0:n*m], tinvn[0:n*m], n_base[0:n])
    if (S->sub_invn != NULL) {
        double *si = S->sub_invn, *cn = S->cell_neff;
        #pragma omp target exit data map(delete: si[0:3*n*nsub], cn[0:n])
    }
    if (S->slot_flux != NULL) {
        double *sf = S->slot_flux, *cp = S->cell_pres, *ph = S->phi;
        #pragma omp target exit data map(delete: sf[0:9*n], cp[0:2*n], ph[0:n])
        free(S->slot_flux); free(S->cell_pres); free(S->phi);
    }
    if (S->tlev != NULL) {
        anuga_int tL = S->tL;
        double *tlev = S->tlev, *tR = S->tR;
        #pragma omp target exit data map(delete: tlev[0:n*tL], tR[0:n*tL*3])
    }
    if (S->ponds != NULL) {
        struct subgrid_ponds *P = S->ponds;
        anuga_int npd = P->npd, nc = P->nc, ntab = P->ntab;
        anuga_int *cells = P->cells, *cstart = P->cstart, *tstart = P->tstart;
        double *tb = P->tb, *tslope = P->tslope, *tvol = P->tvol, *tat = P->tat;
        double *zs = P->zs, *Vzs = P->Vzs, *vmax = P->vmax, *shigh = P->shigh, *zlow = P->zlow, *w = P->w;
        double *V = P->V, *dV = P->dV;
        // no copy back here: Python may already have released V (sync before, with gpu_subgrid_sync_ponds)
        #pragma omp target exit data map(delete: cells[0:nc], cstart[0:nc+1], tstart[0:npd+1])
        #pragma omp target exit data map(delete: tb[0:ntab], tslope[0:ntab], tvol[0:ntab], tat[0:ntab])
        #pragma omp target exit data map(delete: zs[0:npd], Vzs[0:npd], vmax[0:npd], shigh[0:npd], zlow[0:npd], w[0:npd])
        #pragma omp target exit data map(delete: V[0:npd], dV[0:npd])
        free(P);
    }
    free(S);
    GD->subgrid = NULL;
}
