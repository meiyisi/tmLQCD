/***********************************************************************
 *
 * Copyright (C) 2001 Martin Hasenbusch
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
 *
 * File observables.c
 *
 *
 * The externally accessible functions are
 *
 *   double measure_gauge_action(void)
 *     Returns the value of the action
 ************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "sse.h"
#include "su3.h"
#include "su3adj.h"
#include "geometry_eo.h"
#include "global.h"
#include <io/params.h>
#include "measure_gauge_action.h"

double measure_gauge_action(su3 ** const gf) {

/* there is a reduction on ks and kc so for OpenMP these need to be
 * shared variables outside the parallel region
 * we can keep the static keyword and even need it for ga because
 * of the way kc and ks are accessed and the way ga is returned
 * at every call of this function */

  static double ga,ks,kc;
  ks=0.0; kc=0.0;

#ifdef MPI
  static double gas;
#endif
 

#ifdef OMP
#define static
#pragma omp parallel
  {
#endif

  int ix,ix1,ix2,mu1,mu2;
  static su3 pr1,pr2; 
  su3 *v,*w;
  static double ac,tr,ts,tt;

#ifdef OMP
#undef static
#endif

  if(g_update_gauge_energy) {
    kc=0.0; ks=0.0;

#ifdef OMP
#pragma omp for reduction(+:ks) reduction(+:kc)
#endif
    for (ix=0;ix<VOLUME;ix++){
      for (mu1=0;mu1<3;mu1++){ 
	ix1=g_iup[ix][mu1];
	for (mu2=mu1+1;mu2<4;mu2++){ 
	  ix2=g_iup[ix][mu2];
	  v=&gf[ix][mu1];
	  w=&gf[ix1][mu2];
	  _su3_times_su3(pr1,*v,*w);
	  v=&gf[ix][mu2];
	  w=&gf[ix2][mu1];
	  _su3_times_su3(pr2,*v,*w);
	  _trace_su3_times_su3d(ac,pr1,pr2);
	  tr=ac+kc;
	  ts=tr+ks;
	  tt=ts-ks;
	  ks=ts;
	  kc=tr-tt;
	}
      }
    }

#ifdef OMP
  } /* we need to close and reopen the g_update_gauge_energy block 
     * to be able to close the OpenMP parallel section before 
     * doing the final summation */
  } /* OpenMP closing brace */

  if(g_update_gauge_energy) {
#endif

    ga=(kc+ks)/3.0;
#ifdef MPI
    MPI_Allreduce(&ga, &gas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ga = gas;
#endif
    GaugeInfo.plaquetteEnergy = ga;
    g_update_gauge_energy = 0;
  }
  return ga;
}


