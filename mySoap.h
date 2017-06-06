#ifndef MYSOAP/* Include guard */
#define SOAP

#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include <armadillo>
#include <iomanip>
#include "myArmadillo.h"


using namespace std;
using namespace arma;
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
cube getY(int l, vec Theta, vec Phi){
 cube Yn = zeros<cube>(2*l + 1,Theta.n_elem, Phi.n_elem); 
    for(int m=-l; m <= l; m++){ 
     for(int t=0; t < Theta.n_elem; t++){ 
       for(int p=0; p < Phi.n_elem; p++){ 
        Yn(l+m,t,p) = tesseral_spherical_harm(l,m,Theta(t),Phi(p)); 
        }
      }
    }
    return Yn;
};
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
cube getT(int n, int l,int m, mat g, cube Yl, vec R, vec Theta, vec Phi){
 cube T =  zeros<cube>(GL.n_rows,GL.n_rows,GL.n_rows); // (r, Theta, Phi)
  for(int r=0; r < R.n_elem; r++){ 
    for(int t=0; t < Theta.n_elem; t++){ 
      for(int p=0; p < Phi.n_elem; p++){ 
        T(r,t,p) = R(r)* R(r)*sin(Theta(t))*Yl(l+m,t,p); 
      }
    }
  }
  return T;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
cube getTMat(int n, int l,int m, mat g, mat Y00, vec R, vec Theta, vec Phi){
 cube T =  zeros<cube>(GL.n_rows,GL.n_rows,GL.n_rows); // (r, Theta, Phi)
  for(int r=0; r < GL.n_rows; r++){ 
    for(int t=0; t < GL.n_rows; t++){ 
      for(int p=0; p < GL.n_rows; p++){ 
        T(r,t,p) = R(r)* R(r)*sin(Theta(t))*g(n,r)*Y00(t,p); 
      }
    }
  }
  return T;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
double integrate3D(cube rho,cube Tnlm, vec R, vec Theta, vec Phi){

  double x = 0;

  for(int i=0; i < GL.n_rows; i++){
    for(int j=0; j < GL.n_rows; j++){
      for(int k=0; k < GL.n_rows; k++){
     
        x += GL(i,1)*GL(j,1)*GL(k,1)*rho(i,j,k)*Tnlm(i,j,k);

      }
    }
  }
return x;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

#endif 

