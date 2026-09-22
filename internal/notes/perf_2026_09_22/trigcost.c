#include <math.h>
#include <stdio.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(){ static double x[2800], y[2800]; for(int i=0;i<2800;i++) x[i]=0.001*i-1.4;
  double best=1e9; for(int rep=0;rep<5;rep++){double t0=now(); for(int r=0;r<2000;r++){ for(int i=0;i<2000;i+=2){y[i]=sin(x[i]);y[i+1]=cos(x[i+1]);} for(int i=2000;i<2800;i++) y[i]=tanh(x[i]); __asm__ volatile(""::"r"(y):"memory");} double t=(now()-t0)/2000; if(t<best)best=t;}
  printf("2000 sin/cos + 800 tanh scalar: %.2f us  (%.2f ns/call)\n", best*1e6, best*1e9/2800); }
