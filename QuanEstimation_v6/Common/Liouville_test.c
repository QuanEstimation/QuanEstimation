//Liouville_test.c

#include <stdio.h>

#define N  10
#define NC 3
#define Dim  (N+1)*NC


void Liouville_commu(const double (*a)[Dim], int m, double (*b)[Dim*Dim]){
  //  puts("twodemiarr starts!\n");
    int bi, bj, bk, dim, ni, nj, nk;
    dim = m;
    for(bi=0; bi< dim; bi++){
         for(bj=0; bj< dim; bj++){
              for(bk=0; bk< dim; bk++){
                    ni = dim*bi+bj;
                    nj = dim*bk+bj;
                    nk = dim*bi+bk;

                    b[ni][nj] = a[bi][bk];
                    b[ni][nk] = -a[bk][bj];
                    b[ni][ni] = a[bi][bi]-a[bj][bj];
              }
          }
     }
    
  //  puts("twodemiarr ends!\n");
    return;
 }

void Liouville_dissip(const double (*a)[Dim], int m, double (*b)[Dim*Dim]){
    int bi, bj, bk, dim, ni, nj, nk;
    int bl, bp;
    dim = m;
    double L_temp;
    for(bi=0; bi<dim; bi++){
        for(bj=0; bj<dim; bj++){
            ni = dim*bi+bj;
            for(bk=0; bk<dim; bk++){
                for(bl=0; bl<dim; bl++){
                    nj = dim*bk+bl;
                    L_temp = a[bi][bk]*a[bj][bl];
                    for(bp=0; bp<dim; bp++){
                         L_temp = L_temp-0.5*(bk==bi)*a[bp][bj]*a[bp][bl]-0.5*(bl==bj)*a[bp][bk]*a[bp][bi];
                    }
                    b[ni][nj] = L_temp;
                 }
             }
          }

    }
   return;
 }
//     result = np.array(result)
//     result[np.abs(result) < 1e-10] = 0.
//     return result