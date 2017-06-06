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
        Yn.at(l+m,t,p) = tesseral_spherical_harm(l,m,Theta.at(t),Phi.at(p)); 
        }
      }
    }
    return Yn;
};
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
cube getT(int n, int l,int m, mat g, cube Yl, vec R, vec Theta, vec Phi){
 cube T =  zeros<cube>(R.n_elem,Theta.n_elem,Phi.n_elem); // (r, Theta, Phi)
  for(int r=0; r < R.n_elem; r++){ 
    for(int t=0; t < Theta.n_elem; t++){ 
      for(int p=0; p < Phi.n_elem; p++){ 
        T.at(r,t,p) = R.at(r)* R.at(r)*sin(Theta.at(t))*Yl.at(l+m,t,p);  // R*R*sin(Theta) is the Jacobian for the integration.
      }
    }
  }
  return T;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
cube getTMat(int n, int l,int m, mat g, mat Y00, vec R, vec Theta, vec Phi){
 cube T =  zeros<cube>(R.n_elem,Theta.n_elem,Phi.n_elem); // (r, Theta, Phi)

  for(int r=0; r < R.n_elem; r++){ 
    for(int t=0; t < Theta.n_elem; t++){ 
      for(int p=0; p < Phi.n_elem; p++){ 
        T.at(r,t,p) = R.at(r)* R.at(r)*sin(Theta.at(t))*g.at(n,r)*Y00.at(t,p); 
      }
    }
  }
  return T;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
double integrate3D(cube rho,cube Tnlm, cube GLC, vec R, vec Theta, vec Phi){

  double x = 0;

  for(int i=0; i < GLC.n_slices; i++){
    for(int j=0; j < GLC.n_slices; j++){
      for(int k=0; k < GLC.n_slices; k++){
     
        x += GLC.at(i,j,k)*rho.at(i,j,k)*Tnlm.at(i,j,k); // DANGER!!! NOT RESCALED -> MUST BE RESCALED BY 0.5*0.5*0.5*pi*pi*rcut in main.
                                                             // This is to increase the computation time.

      }
    }
  }
return x;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

#endif 

