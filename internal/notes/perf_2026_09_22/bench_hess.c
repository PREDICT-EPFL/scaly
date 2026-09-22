#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "race_car_closed_loop_N200_hess_lower.h"
#define NNZ 2607
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static double urand(){return rand()/(double)RAND_MAX;}
int FN(const double**,double**,int*,double*,int);
int main(int argc,char**argv){
  int mem = argc>1?atoi(argv[1]):0;
  static double z[1206],p[811],sig[1],lam[1204],out[NNZ],w[36084];
  srand(1);
  for(int k=0;k<201;k++){z[6*k]=urand()*10;z[6*k+1]=urand()*10;z[6*k+2]=urand()*6-3;z[6*k+3]=urand()*20;z[6*k+4]=urand()*2-1;z[6*k+5]=urand()*0.8-0.4;}
  for(int k=0;k<201;k++){p[4*k]=urand();p[4*k+1]=urand();p[4*k+2]=urand()*6-3;p[4*k+3]=urand()*20;}
  double par[7]={2.5,0.05,200,1000,10,1,0.5}; for(int i=0;i<7;i++)p[804+i]=par[i];
  sig[0]=0.7; for(int i=0;i<1204;i++)lam[i]=urand()*2-1;
  const double* arg[4]={z,p,sig,lam}; double* res[1]={out};
  FN(arg,res,NULL,w,mem);
  if(argc>2){FILE*f=fopen(argv[2],"wb");fwrite(out,8,NNZ,f);fclose(f);}
  int R=20000; double best=1e9;
  for(int rep=0;rep<5;rep++){double t0=now();for(int r=0;r<R;r++){FN(arg,res,NULL,w,mem);__asm__ volatile(""::"r"(out):"memory");}double t=(now()-t0)/R; if(t<best)best=t;}
  printf("mem=%d  %.2f us/call\n",mem,best*1e6);
}
