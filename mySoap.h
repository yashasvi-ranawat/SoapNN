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
 vec RR = R%R;
 vec sinT = sin(Theta);
  for(int r=0; r < R.n_elem; r++){ 
    for(int t=0; t < Theta.n_elem; t++){ 
      for(int p=0; p < Phi.n_elem; p++){ 
        T.at(r,t,p) = g.at(n,r)*RR.at(r)*sinT.at(t)*Yl.at(l+m,t,p);  // R*R*sin(Theta) is the Jacobian for the integration.
      }
    }
  }
  return T;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//cube getT(int n, int l,int m, mat g, cube Yl, vec R, vec Theta, vec Phi){
// cube T =  zeros<cube>(R.n_elem,Theta.n_elem,Phi.n_elem); // (r, Theta, Phi)
//  for(int r=0; r < R.n_elem; r++){ 
//    for(int t=0; t < Theta.n_elem; t++){ 
//      for(int p=0; p < Phi.n_elem; p++){ 
//        T.at(r,t,p) = g.at(n,r)*R.at(r)* R.at(r)*sin(Theta.at(t))*Yl.at(l+m,t,p);  // R*R*sin(Theta) is the Jacobian for the integration.
//      }
//    }
//  }
//
//  return T;
//}
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
cube getGaussDistr( mat coord,vec R, vec The, vec Phi, cube X, cube Y , cube Z, double sig){
//!!!! DOUNT FORGET IN MAIN TO DO:
//  sig = 1/sig;
//  sig = sig*sig;
//  sig = 0.5*sig;
//  DONT FORGET TO RESCALE ORIGIN FOR GAUSS-LEGENDRE QUADUATURE!
  
  cube G(R.n_elem,The.n_elem,Phi.n_elem);
  double x1, x2, x3; // X1 = R*sin(Theta)*cos(Phi), X2 = R*sin(Theta)*sin(Phi), X3 = R*cos(Theta) 
  rowvec origin = coord.row(coord.n_rows - 1); // last xyz position of the .xyz file.

  for(int i=0; i < coord.n_rows; i++){
  coord.row(i) = coord.row(i) - origin; // shifting all xyz position so that H is on the origin. 
  }

  for(int p=0; p < coord.n_rows - 1; p++)
    for(int i=0; i < R.n_rows; i++)
      for(int j=0; j < The.n_rows; j++)
        for(int k=0; k < Phi.n_rows; k++){ 
          x1 = X.at(i,j,k);
          x2 = Y.at(i,j,k);
          x3 = Z.at(i,j,k);
//          G(i,j,k) = G(i,j,k) + exp(-abs((coord(p,0) - x1)) - abs((coord(p,1) - x2)) - abs((coord(p,2) - x3)));
          G.at(i,j,k) = G.at(i,j,k) + exp(-pow(((coord(p,0) - x1)),2) - pow((coord(p,1) - x2),2) - pow((coord(p,2) - x3),2)/1.0);
         }

 return G;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
double integ3D(cube intMea,cube Tnlm){

  // DANGER!!! NOT RESCALED -> MUST BE RESCALED BY 0.5*0.5*0.5*pi*pi*rcut*integ3D() in main.
  // This is to increase the computation time.
  double z;

  mat x1 = sum(intMea%Tnlm);
  rowvec y1 = sum(x1);

  z = sum(y1);

return z;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
vec integ3Dvec(cube intMea,cube intMeb, cube intAll ,cube Tnlm){

  // DANGER!!! NOT RESCALED -> MUST BE RESCALED BY 0.5*0.5*0.5*pi*pi*rcut*integ3D() in main.
  // This is to increase the computation time.
  vec z(3);

  mat x1 = sum(intMea%Tnlm);
  mat x2 = sum(intMeb%Tnlm);
  mat x3 = sum(intAll%Tnlm);

  rowvec y1 = sum(x1);
  rowvec y2 = sum(x2);
  rowvec y3 = sum(x3);

  z(0) = sum(y1);
  z(1) = sum(y2);
  z(2) = sum(y3);

return z;
}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

#endif 

