/* time only the 200 stage kernels of the original, scalar noinline */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void race_car_eq_interstage_adj0_0_1_fwd4cb455f2851e_adj_eq_z_hoisted_1_raw(const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, double*, double*);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(){ static double z[1206], lam[1204], o[9600]; double P[7]={2.5,0.05,200,1000,10,1,0.5}, t2=P[1]/6, t7=P[1]/2, t28=0.5*P[0], t69=1/t28, t93=1/P[2], t895[4]={0,0,0,t7*P[3]*t93};
  srand(1); for(int i=0;i<1206;i++) z[i]=rand()/(double)RAND_MAX; for(int i=0;i<1204;i++) lam[i]=rand()/(double)RAND_MAX*2-1;
  double best=1e9; for(int rep=0;rep<5;rep++){double t0=now(); for(int r=0;r<20000;r++){ for(int k=0;k<200;k++) race_car_eq_interstage_adj0_0_1_fwd4cb455f2851e_adj_eq_z_hoisted_1_raw(z+6*k,P,lam+4+4*k,&t2,&t28,&t69,&t7,t895,&t93,o+48*k,0); __asm__ volatile(""::"r"(o):"memory");} double t=(now()-t0)/20000; if(t<best)best=t;}
  printf("200 original stage kernels: %.2f us\n", best*1e6); }
