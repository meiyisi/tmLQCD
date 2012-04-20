/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#ifdef MPI
#include <mpi.h>
#endif
#ifdef OMP
# include <omp.h>
#endif
#include "su3.h"
#include "scalar_prod.h"

/*  <S,R>=S^* times R */
_Complex double scalar_prod(spinor * const S, spinor * const R, const int N, const int parallel){
#ifdef OMP
#define static
#endif

  static _Complex double ks,kc;
  ks = kc = 0.0;
  _Complex double c;

#ifdef MPI
  _Complex double d;
#endif
  

#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix;
  static _Complex double ds,tr,ts,tt;
  spinor *s,*r;
  
#if (defined BGL && defined XLC)
  __alignx(16, S);
  __alignx(16, R);
#endif
#ifdef OMP
#undef static
#pragma omp for reduction(+:kc) reduction(+:ks)
#endif
  for (ix = 0; ix < N; ix++)
  {
    s=(spinor *) S + ix;
    r=(spinor *) R + ix;
    
    ds = r->s0.c0 * conj(s->s0.c0) + r->s0.c1 * conj(s->s0.c1) + r->s0.c2 * conj(s->s0.c2) +
         r->s1.c0 * conj(s->s1.c0) + r->s1.c1 * conj(s->s1.c1) + r->s1.c2 * conj(s->s1.c2) +
	 r->s2.c0 * conj(s->s2.c0) + r->s2.c1 * conj(s->s2.c1) + r->s2.c2 * conj(s->s2.c2) + 
         r->s3.c0 * conj(s->s3.c0) + r->s3.c1 * conj(s->s3.c1) + r->s3.c2 * conj(s->s3.c2);

    /* Kahan Summation */    
    tr=ds+kc;
    ts=tr+ks;
    tt=ts-ks;
    ks=ts;
    kc=tr-tt;
  }

#ifdef OMP
  } /* OpenMP closing brace */
#endif

  kc=ks+kc;

  c = kc;

#ifdef MPI
  if(parallel == 1)
  {
    d = c;
    MPI_Allreduce(&d, &c, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return(c);
}

#ifdef WITHLAPH
_Complex double scalar_prod_su3vect(su3_vector * const S, su3_vector * const R, const int N, const int parallel)
{
#ifdef OMP
#define static
#endif

  static double ks;
  _Complex double c;

  ks = c = 0.0;

#ifdef MPI
  _Complex double d;
#endif


#ifdef OMP
#pragma omp parallel
  {
#endif

  static double ds, tr, ts, tt;
  su3_vector *s, *r;

  /* Real Part */
#ifdef OMP
#undef static
#pragma omp for reduction(+:c) reduction(+:ks)
#endif
  for (int ix = 0; ix < N; ++ix)
    {
      s = (su3_vector *) S + ix;
      r = (su3_vector *) R + ix;
    
      ds = r->c0 * conj(s->c0) + r->c1 * conj(s->c1) + r->c2 * conj(s->c2);

      /* Kahan Summation */    
      tr = ds + c;
      ts = tr + ks;
      tt = ts - ks;
      ks = ts;
      c  = tr - tt;
    }
#ifdef OMP
  } /* OpenMP closing brace */
#endif

  c = ks + c;

#ifdef MPI
  if(parallel == 1)
  {
    d = c;
    MPI_Allreduce(&d, &c, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return(c);
}
#endif
