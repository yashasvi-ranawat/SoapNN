#include<iostream>
#include<armadillo>
#include"myArmadillo.h"
#include"myMath.h"
#include"mySoap.h"

using namespace std;
using namespace arma;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is a SOAP algorithm with real spherical harmonics with the radial basis function of polynomials larger than //
// order of 2. The algorithm is based on [On representing chemical environments by Albert P. Batrok et al.] and     // 
// [Comparing molecules and solids across structural and alchemical space by Sandip De et al.]. In the cose, the    //
// equations will be reffered as [APB eq.#] and [SD eq.#]. The armadillo package [http://arma.sourceforge.net/]  is //
// heavily used due to its Octave/Matlab/Numpy like syntax which accelerates the developement speed.                //
//   The aim is to attempt to use the power spectrum for Neural Network instead for KRR.                            //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // The algorithm is devided in 5 parts:                                                                          //
  // 1) Retrieving data. -> W[APB eq.25], Gauss-Legendre values for integration,                                   //
  // 2) Constructing Basis functions -> g_n(r)[APB eq.25], Y_lm(Theta,Phi)[Tesseral Harmonics]                     //
  // 3) Preperation for integration. -> T[n][l][m][r][Theta][Phi] = r^2 sin(Theta) g_n(r) Y_lm(Theta,Phi).         //
  //                                     ( Tensor that is independent of the inputs.)                              //
  // 4) Get coeffs c_nlm[APD eq.24] = Integrate Rho(r,Theta,Phi) T dV, where Rho[SD eq.14] is the Gaussian smeared //
  //                                    xyz atomic positions.                                                      //
  // 5) Get power spectrum by P_b1b2lab = sum_m (c_b1lma  c_b2lmb) [SD eq.17]                                      //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //----------------------------------------------------------------------------------------------------------------
  // Part 0) Setting Parameters
  //----------------------------------------------------------------------------------------------------------------
  int lMax = 9;
  
  double pi = 3.14159265358979324;
  double halfPi = 3.14159265358979324*0.5;


  double rcut = 3.6;
  double rescaleInt = pi*pi*0.5*0.5*rcut;

  cube rho = ones<cube>(100,100,100);

  
  //----------------------------------------------------------------------------------------------------------------
  // Part 1) Retrieving Data -> W, Gauss-Legendre, XYZ-Smeared
  //----------------------------------------------------------------------------------------------------------------
  
  mat W; // [APB eq.25]
  W.load("WMatData/W5.csv");

  mat GL; // [http://keisan.casio.com/exec/system/1329114617 (June 5th 2017)] , produced by Octave. W(:,0) -> GL coord. pos. W(:,1) -> GL weights.
  GL.load("parameters100.txt");

  vec R = rcut*0.5*GL.col(0) + rcut*0.5 ; // rescaled radius for gauss-legendre quaduature
  vec Theta = pi*GL.col(0)*0.5 + pi*0.5;  // rescaled radius for gauss-legendre quaduature
  vec Phi = pi*GL.col(0) + pi;            // rescaled radius for gauss-legendre quaduature
  R.print(" ");

  mat coord = getPos(argv[1]);
  string* type = getType(argv[1]);

  cube GLC(GL.n_rows,GL.n_rows,GL.n_rows);
  for(int i=0; i < GL.n_rows; i++){ 
   for(int j=0; j < GL.n_rows; j++){ 
    for(int k=0; k < GL.n_rows; k++){ 
      

      GLC.at(i,j,k) = GL.at(i,1)*GL.at(j,1)*GL.at(k,1);

      }
    }
  }


  //----------------------------------------------------------------------------------------------------------------
  // Part 2) Constructing Basis Functions -> g_n(r), N_a, phi_a(r),  
  //----------------------------------------------------------------------------------------------------------------

  
  vec N = zeros<vec>(5); //[APB eq.25]

  for(int i=0; i < N.n_rows; i++){ 
    N(i) = sqrt(pow(rcut, 2*(i + 1) + 5)/(2*(i + 1) + 5)); // DO I NEED TO RESCAEL rcut???
   }

  mat phi = zeros<mat>(5,GL.n_rows); //[APB eq.25]
  for(int r=0; r < GL.n_rows; r++){ 
    for(int a=0; a < N.n_rows; a++){ 
//      phi(a,r) = pow(rcut - GL(r,0), 2*(a + 3))/N(a); // not rescaled
      phi(a,r) = pow(rcut - R(r), 2*(a + 3))/N(a); // rescaled
    }
  }


  mat g = zeros<mat>(N.n_rows,GL.n_rows); // Radial Basis Functions [APB eq.25]. g(*,:) -> n's. g(:,*) -> r's of GL coord. pos.
  for(int r=0; r < GL.n_rows; r++){ 
    for(int n=0; n < N.n_rows; n++){ 
      for(int a=0; a < W.n_rows; a++){ 
        g.at(n,r) = g.at(n,r) + W.at(n,a)*phi.at(a,r); // DO I NEED TO RESCALE W???
      }
    }
  }


  mat Y0 = zeros<mat>(GL.n_rows, GL.n_rows);// Tesseral Spherical Harmonics at GL coord. pos. first lMax is l, second lMax is m but patted with 0's. 
  for(int t=0; t < GL.n_rows; t++){ 
    for(int p=0; p < GL.n_rows; p++){ 
  //    Y0(t,p) = tesseral_spherical_harm(0,0,GL(t,0),GL(p,0)); // not rescaled.
      Y0.at(t,p) = tesseral_spherical_harm(0,0,Theta.at(t),Phi.at(p)); // rescaled
      }
    }

  cube Y1 = getY(1,Theta, Phi);
  cube Y2 = getY(2,Theta, Phi);
  cube Y3 = getY(3,Theta, Phi);
  cube Y4 = getY(4,Theta, Phi);
  cube Y5 = getY(5,Theta, Phi);
  cube Y6 = getY(6,Theta, Phi);
  cube Y7 = getY(7,Theta, Phi);
//  cube Y8 = getY(8,Theta, Phi);
//  cube Y9 = getY(9,Theta, Phi);
   
  //----------------------------------------------------------------------------------------------------------------
  // Part 3) Preparing Integrand -> Cube Tnlm(r,Theta,Phi)
  //----------------------------------------------------------------------------------------------------------------

  cube T000 = getTMat(0,0,0,g,Y0,R,Theta,Phi); cube T100 = getTMat(1,0,0,g,Y0,R,Theta,Phi);  cube T200 = getTMat(2,0,0,g,Y0,R,Theta,Phi);
  cube T01m1 = getT(0,1,-1,g,Y1,R,Theta,Phi);  cube T11m1 = getT(1,1,-1,g,Y1,R,Theta,Phi);  cube T21m1 = getT(2,1,-1,g,Y1,R,Theta,Phi);
  cube T010  = getT(0,1,+0,g,Y1,R,Theta,Phi);  cube T110  = getT(1,1,+0,g,Y1,R,Theta,Phi);  cube T210  = getT(2,1,+0,g,Y1,R,Theta,Phi);
  cube T01p1 = getT(0,1,+1,g,Y1,R,Theta,Phi);  cube T11p1 = getT(1,1,+1,g,Y1,R,Theta,Phi);  cube T21p1 = getT(2,1,+1,g,Y1,R,Theta,Phi);
  cube T02m2 = getT(0,2,-2,g,Y2,R,Theta,Phi);  cube T12m2 = getT(1,2,-2,g,Y2,R,Theta,Phi);  cube T22m2 = getT(2,2,-2,g,Y2,R,Theta,Phi);
  cube T02m1 = getT(0,2,-1,g,Y2,R,Theta,Phi);  cube T12m1 = getT(1,2,-1,g,Y2,R,Theta,Phi);  cube T22m1 = getT(2,2,-1,g,Y2,R,Theta,Phi);
  cube T020  = getT(0,2,+0,g,Y2,R,Theta,Phi);  cube T120  = getT(1,2,+0,g,Y2,R,Theta,Phi);  cube T220  = getT(2,2,+0,g,Y2,R,Theta,Phi);
  cube T02p1 = getT(0,2,+1,g,Y2,R,Theta,Phi);  cube T12p1 = getT(1,2,+1,g,Y2,R,Theta,Phi);  cube T22p1 = getT(2,2,+1,g,Y2,R,Theta,Phi);
  cube T02p2 = getT(0,2,+2,g,Y2,R,Theta,Phi);  cube T12p2 = getT(1,2,+2,g,Y2,R,Theta,Phi);  cube T22p2 = getT(2,2,+2,g,Y2,R,Theta,Phi);
  cube T03m3 = getT(0,3,-3,g,Y3,R,Theta,Phi);  cube T13m3 = getT(1,3,-3,g,Y3,R,Theta,Phi);  cube T23m3 = getT(2,3,-3,g,Y3,R,Theta,Phi);
  cube T03m2 = getT(0,3,-2,g,Y3,R,Theta,Phi);  cube T13m2 = getT(1,3,-2,g,Y3,R,Theta,Phi);  cube T23m2 = getT(2,3,-2,g,Y3,R,Theta,Phi);
  cube T03m1 = getT(0,3,-1,g,Y3,R,Theta,Phi);  cube T13m1 = getT(1,3,-1,g,Y3,R,Theta,Phi);  cube T23m1 = getT(2,3,-1,g,Y3,R,Theta,Phi);
  cube T030  = getT(0,3,+0,g,Y3,R,Theta,Phi);  cube T130  = getT(1,3,+0,g,Y3,R,Theta,Phi);  cube T230  = getT(2,3,+0,g,Y3,R,Theta,Phi);
  cube T03p1 = getT(0,3,+1,g,Y3,R,Theta,Phi);  cube T13p1 = getT(1,3,+1,g,Y3,R,Theta,Phi);  cube T23p1 = getT(2,3,+1,g,Y3,R,Theta,Phi);
  cube T03p2 = getT(0,3,+2,g,Y3,R,Theta,Phi);  cube T13p2 = getT(1,3,+2,g,Y3,R,Theta,Phi);  cube T23p2 = getT(2,3,+2,g,Y3,R,Theta,Phi);
  cube T03p3 = getT(0,3,+3,g,Y3,R,Theta,Phi);  cube T13p3 = getT(1,3,+3,g,Y3,R,Theta,Phi);  cube T23p3 = getT(2,3,+3,g,Y3,R,Theta,Phi);
  cube T04m4 = getT(0,4,-4,g,Y4,R,Theta,Phi);  cube T14m4 = getT(1,4,-4,g,Y4,R,Theta,Phi);  cube T24m4 = getT(2,4,-4,g,Y4,R,Theta,Phi);
  cube T04m3 = getT(0,4,-3,g,Y4,R,Theta,Phi);  cube T14m3 = getT(1,4,-3,g,Y4,R,Theta,Phi);  cube T24m3 = getT(2,4,-3,g,Y4,R,Theta,Phi);
  cube T04m2 = getT(0,4,-2,g,Y4,R,Theta,Phi);  cube T14m2 = getT(1,4,-2,g,Y4,R,Theta,Phi);  cube T24m2 = getT(2,4,-2,g,Y4,R,Theta,Phi);
  cube T04m1 = getT(0,4,-1,g,Y4,R,Theta,Phi);  cube T14m1 = getT(1,4,-1,g,Y4,R,Theta,Phi);  cube T24m1 = getT(2,4,-1,g,Y4,R,Theta,Phi);
  cube T040  = getT(0,4,+0,g,Y4,R,Theta,Phi);  cube T140  = getT(1,4,+0,g,Y4,R,Theta,Phi);  cube T240  = getT(2,4,+0,g,Y4,R,Theta,Phi);
  cube T04p1 = getT(0,4,+1,g,Y4,R,Theta,Phi);  cube T14p1 = getT(1,4,+1,g,Y4,R,Theta,Phi);  cube T24p1 = getT(2,4,+1,g,Y4,R,Theta,Phi);
  cube T04p2 = getT(0,4,+2,g,Y4,R,Theta,Phi);  cube T14p2 = getT(1,4,+2,g,Y4,R,Theta,Phi);  cube T24p2 = getT(2,4,+2,g,Y4,R,Theta,Phi);
  cube T04p3 = getT(0,4,+3,g,Y4,R,Theta,Phi);  cube T14p3 = getT(1,4,+3,g,Y4,R,Theta,Phi);  cube T24p3 = getT(2,4,+3,g,Y4,R,Theta,Phi);
  cube T04p4 = getT(0,4,+4,g,Y4,R,Theta,Phi);  cube T14p4 = getT(1,4,+4,g,Y4,R,Theta,Phi);  cube T24p4 = getT(2,4,+4,g,Y4,R,Theta,Phi);
  cube T05m5 = getT(0,5,-5,g,Y5,R,Theta,Phi);  cube T15m5 = getT(1,5,-5,g,Y5,R,Theta,Phi);  cube T25m5 = getT(2,5,-5,g,Y5,R,Theta,Phi);
  cube T05m4 = getT(0,5,-4,g,Y5,R,Theta,Phi);  cube T15m4 = getT(1,5,-4,g,Y5,R,Theta,Phi);  cube T25m4 = getT(2,5,-4,g,Y5,R,Theta,Phi);
  cube T05m3 = getT(0,5,-3,g,Y5,R,Theta,Phi);  cube T15m3 = getT(1,5,-3,g,Y5,R,Theta,Phi);  cube T25m3 = getT(2,5,-3,g,Y5,R,Theta,Phi);
  cube T05m2 = getT(0,5,-2,g,Y5,R,Theta,Phi);  cube T15m2 = getT(1,5,-2,g,Y5,R,Theta,Phi);  cube T25m2 = getT(2,5,-2,g,Y5,R,Theta,Phi);
  cube T05m1 = getT(0,5,-1,g,Y5,R,Theta,Phi);  cube T15m1 = getT(1,5,-1,g,Y5,R,Theta,Phi);  cube T25m1 = getT(2,5,-1,g,Y5,R,Theta,Phi);
  cube T050  = getT(0,5,+0,g,Y5,R,Theta,Phi);  cube T150  = getT(1,5,+0,g,Y5,R,Theta,Phi);  cube T250  = getT(2,5,+0,g,Y5,R,Theta,Phi);
  cube T05p1 = getT(0,5,+1,g,Y5,R,Theta,Phi);  cube T15p1 = getT(1,5,+1,g,Y5,R,Theta,Phi);  cube T25p1 = getT(2,5,+1,g,Y5,R,Theta,Phi);
  cube T05p2 = getT(0,5,+2,g,Y5,R,Theta,Phi);  cube T15p2 = getT(1,5,+2,g,Y5,R,Theta,Phi);  cube T25p2 = getT(2,5,+2,g,Y5,R,Theta,Phi);
  cube T05p3 = getT(0,5,+3,g,Y5,R,Theta,Phi);  cube T15p3 = getT(1,5,+3,g,Y5,R,Theta,Phi);  cube T25p3 = getT(2,5,+3,g,Y5,R,Theta,Phi);
  cube T05p4 = getT(0,5,+4,g,Y5,R,Theta,Phi);  cube T15p4 = getT(1,5,+4,g,Y5,R,Theta,Phi);  cube T25p4 = getT(2,5,+4,g,Y5,R,Theta,Phi);
  cube T05p5 = getT(0,5,+5,g,Y5,R,Theta,Phi);  cube T15p5 = getT(1,5,+5,g,Y5,R,Theta,Phi);  cube T25p5 = getT(2,5,+5,g,Y5,R,Theta,Phi);
  cube T06m6 = getT(0,6,-6,g,Y6,R,Theta,Phi);  cube T16m6 = getT(1,6,-6,g,Y6,R,Theta,Phi);  cube T26m6 = getT(2,6,-6,g,Y6,R,Theta,Phi);
  cube T06m5 = getT(0,6,-5,g,Y6,R,Theta,Phi);  cube T16m5 = getT(1,6,-5,g,Y6,R,Theta,Phi);  cube T26m5 = getT(2,6,-5,g,Y6,R,Theta,Phi);
  cube T06m4 = getT(0,6,-4,g,Y6,R,Theta,Phi);  cube T16m4 = getT(1,6,-4,g,Y6,R,Theta,Phi);  cube T26m4 = getT(2,6,-4,g,Y6,R,Theta,Phi);
  cube T06m3 = getT(0,6,-3,g,Y6,R,Theta,Phi);  cube T16m3 = getT(1,6,-3,g,Y6,R,Theta,Phi);  cube T26m3 = getT(2,6,-3,g,Y6,R,Theta,Phi);
  cube T06m2 = getT(0,6,-2,g,Y6,R,Theta,Phi);  cube T16m2 = getT(1,6,-2,g,Y6,R,Theta,Phi);  cube T26m2 = getT(2,6,-2,g,Y6,R,Theta,Phi);
  cube T06m1 = getT(0,6,-1,g,Y6,R,Theta,Phi);  cube T16m1 = getT(1,6,-1,g,Y6,R,Theta,Phi);  cube T26m1 = getT(2,6,-1,g,Y6,R,Theta,Phi);
  cube T060  = getT(0,6,+0,g,Y6,R,Theta,Phi);  cube T160  = getT(1,6,+0,g,Y6,R,Theta,Phi);  cube T260  = getT(2,6,+0,g,Y6,R,Theta,Phi);
  cube T06p1 = getT(0,6,+1,g,Y6,R,Theta,Phi);  cube T16p1 = getT(1,6,+1,g,Y6,R,Theta,Phi);  cube T26p1 = getT(2,6,+1,g,Y6,R,Theta,Phi);
  cube T06p2 = getT(0,6,+2,g,Y6,R,Theta,Phi);  cube T16p2 = getT(1,6,+2,g,Y6,R,Theta,Phi);  cube T26p2 = getT(2,6,+2,g,Y6,R,Theta,Phi);
  cube T06p3 = getT(0,6,+3,g,Y6,R,Theta,Phi);  cube T16p3 = getT(1,6,+3,g,Y6,R,Theta,Phi);  cube T26p3 = getT(2,6,+3,g,Y6,R,Theta,Phi);
  cube T06p4 = getT(0,6,+4,g,Y6,R,Theta,Phi);  cube T16p4 = getT(1,6,+4,g,Y6,R,Theta,Phi);  cube T26p4 = getT(2,6,+4,g,Y6,R,Theta,Phi);
  cube T06p5 = getT(0,6,+5,g,Y6,R,Theta,Phi);  cube T16p5 = getT(1,6,+5,g,Y6,R,Theta,Phi);  cube T26p5 = getT(2,6,+5,g,Y6,R,Theta,Phi);
  cube T06p6 = getT(0,6,+6,g,Y6,R,Theta,Phi);  cube T16p6 = getT(1,6,+6,g,Y6,R,Theta,Phi);  cube T26p6 = getT(2,6,+6,g,Y6,R,Theta,Phi);
  cube T07m7 = getT(0,7,-7,g,Y7,R,Theta,Phi);  cube T17m7 = getT(1,7,-7,g,Y7,R,Theta,Phi);  cube T27m7 = getT(2,7,-7,g,Y7,R,Theta,Phi);
  cube T07m6 = getT(0,7,-6,g,Y7,R,Theta,Phi);  cube T17m6 = getT(1,7,-6,g,Y7,R,Theta,Phi);  cube T27m6 = getT(2,7,-6,g,Y7,R,Theta,Phi);
  cube T07m5 = getT(0,7,-5,g,Y7,R,Theta,Phi);  cube T17m5 = getT(1,7,-5,g,Y7,R,Theta,Phi);  cube T27m5 = getT(2,7,-5,g,Y7,R,Theta,Phi);
  cube T07m4 = getT(0,7,-4,g,Y7,R,Theta,Phi);  cube T17m4 = getT(1,7,-4,g,Y7,R,Theta,Phi);  cube T27m4 = getT(2,7,-4,g,Y7,R,Theta,Phi);
  cube T07m3 = getT(0,7,-3,g,Y7,R,Theta,Phi);  cube T17m3 = getT(1,7,-3,g,Y7,R,Theta,Phi);  cube T27m3 = getT(2,7,-3,g,Y7,R,Theta,Phi);
  cube T07m2 = getT(0,7,-2,g,Y7,R,Theta,Phi);  cube T17m2 = getT(1,7,-2,g,Y7,R,Theta,Phi);  cube T27m2 = getT(2,7,-2,g,Y7,R,Theta,Phi);
  cube T07m1 = getT(0,7,-1,g,Y7,R,Theta,Phi);  cube T17m1 = getT(1,7,-1,g,Y7,R,Theta,Phi);  cube T27m1 = getT(2,7,-1,g,Y7,R,Theta,Phi);
  cube T070  = getT(0,7,+0,g,Y7,R,Theta,Phi);  cube T170  = getT(1,7,+0,g,Y7,R,Theta,Phi);  cube T270  = getT(2,7,+0,g,Y7,R,Theta,Phi);
  cube T07p1 = getT(0,7,+1,g,Y7,R,Theta,Phi);  cube T17p1 = getT(1,7,+1,g,Y7,R,Theta,Phi);  cube T27p1 = getT(2,7,+1,g,Y7,R,Theta,Phi);
  cube T07p2 = getT(0,7,+2,g,Y7,R,Theta,Phi);  cube T17p2 = getT(1,7,+2,g,Y7,R,Theta,Phi);  cube T27p2 = getT(2,7,+2,g,Y7,R,Theta,Phi);
  cube T07p3 = getT(0,7,+3,g,Y7,R,Theta,Phi);  cube T17p3 = getT(1,7,+3,g,Y7,R,Theta,Phi);  cube T27p3 = getT(2,7,+3,g,Y7,R,Theta,Phi);
  cube T07p4 = getT(0,7,+4,g,Y7,R,Theta,Phi);  cube T17p4 = getT(1,7,+4,g,Y7,R,Theta,Phi);  cube T27p4 = getT(2,7,+4,g,Y7,R,Theta,Phi);
  cube T07p5 = getT(0,7,+5,g,Y7,R,Theta,Phi);  cube T17p5 = getT(1,7,+5,g,Y7,R,Theta,Phi);  cube T27p5 = getT(2,7,+5,g,Y7,R,Theta,Phi);
  cube T07p6 = getT(0,7,+6,g,Y7,R,Theta,Phi);  cube T17p6 = getT(1,7,+6,g,Y7,R,Theta,Phi);  cube T27p6 = getT(2,7,+6,g,Y7,R,Theta,Phi);
  cube T07p7 = getT(0,7,+7,g,Y7,R,Theta,Phi);  cube T17p7 = getT(1,7,+7,g,Y7,R,Theta,Phi);  cube T27p7 = getT(2,7,+7,g,Y7,R,Theta,Phi);
//  cube T08m8 = getT(0,8,-8,g,Y8,R,Theta,Phi);  cube T18m8 = getT(1,8,-8,g,Y8,R,Theta,Phi);  cube T28m8 = getT(2,8,-8,g,Y8,R,Theta,Phi);
//  cube T08m7 = getT(0,8,-7,g,Y8,R,Theta,Phi);  cube T18m7 = getT(1,8,-7,g,Y8,R,Theta,Phi);  cube T28m7 = getT(2,8,-7,g,Y8,R,Theta,Phi);
//  cube T08m6 = getT(0,8,-6,g,Y8,R,Theta,Phi);  cube T18m6 = getT(1,8,-6,g,Y8,R,Theta,Phi);  cube T28m6 = getT(2,8,-6,g,Y8,R,Theta,Phi);
//  cube T08m5 = getT(0,8,-5,g,Y8,R,Theta,Phi);  cube T18m5 = getT(1,8,-5,g,Y8,R,Theta,Phi);  cube T28m5 = getT(2,8,-5,g,Y8,R,Theta,Phi);
//  cube T08m4 = getT(0,8,-4,g,Y8,R,Theta,Phi);  cube T18m4 = getT(1,8,-4,g,Y8,R,Theta,Phi);  cube T28m4 = getT(2,8,-4,g,Y8,R,Theta,Phi);
//  cube T08m3 = getT(0,8,-3,g,Y8,R,Theta,Phi);  cube T18m3 = getT(1,8,-3,g,Y8,R,Theta,Phi);  cube T28m3 = getT(2,8,-3,g,Y8,R,Theta,Phi);
//  cube T08m2 = getT(0,8,-2,g,Y8,R,Theta,Phi);  cube T18m2 = getT(1,8,-2,g,Y8,R,Theta,Phi);  cube T28m2 = getT(2,8,-2,g,Y8,R,Theta,Phi);
//  cube T08m1 = getT(0,8,-1,g,Y8,R,Theta,Phi);  cube T18m1 = getT(1,8,-1,g,Y8,R,Theta,Phi);  cube T28m1 = getT(2,8,-1,g,Y8,R,Theta,Phi);
//  cube T080  = getT(0,8,+0,g,Y8,R,Theta,Phi);  cube T180  = getT(1,8,+0,g,Y8,R,Theta,Phi);  cube T280  = getT(2,8,+0,g,Y8,R,Theta,Phi);
//  cube T08p1 = getT(0,8,+1,g,Y8,R,Theta,Phi);  cube T18p1 = getT(1,8,+1,g,Y8,R,Theta,Phi);  cube T28p1 = getT(2,8,+1,g,Y8,R,Theta,Phi);
//  cube T08p2 = getT(0,8,+2,g,Y8,R,Theta,Phi);  cube T18p2 = getT(1,8,+2,g,Y8,R,Theta,Phi);  cube T28p2 = getT(2,8,+2,g,Y8,R,Theta,Phi);
//  cube T08p3 = getT(0,8,+3,g,Y8,R,Theta,Phi);  cube T18p3 = getT(1,8,+3,g,Y8,R,Theta,Phi);  cube T28p3 = getT(2,8,+3,g,Y8,R,Theta,Phi);
//  cube T08p4 = getT(0,8,+4,g,Y8,R,Theta,Phi);  cube T18p4 = getT(1,8,+4,g,Y8,R,Theta,Phi);  cube T28p4 = getT(2,8,+4,g,Y8,R,Theta,Phi);
//  cube T08p5 = getT(0,8,+5,g,Y8,R,Theta,Phi);  cube T18p5 = getT(1,8,+5,g,Y8,R,Theta,Phi);  cube T28p5 = getT(2,8,+5,g,Y8,R,Theta,Phi);
//  cube T08p6 = getT(0,8,+6,g,Y8,R,Theta,Phi);  cube T18p6 = getT(1,8,+6,g,Y8,R,Theta,Phi);  cube T28p6 = getT(2,8,+6,g,Y8,R,Theta,Phi);
//  cube T08p7 = getT(0,8,+7,g,Y8,R,Theta,Phi);  cube T18p7 = getT(1,8,+7,g,Y8,R,Theta,Phi);  cube T28p7 = getT(2,8,+7,g,Y8,R,Theta,Phi);
//  cube T08p8 = getT(0,8,+8,g,Y8,R,Theta,Phi);  cube T18p8 = getT(1,8,+8,g,Y8,R,Theta,Phi);  cube T28p8 = getT(2,8,+8,g,Y8,R,Theta,Phi);
//  cube T09m9 = getT(0,9,-9,g,Y9,R,Theta,Phi);  cube T19m9 = getT(1,9,-9,g,Y9,R,Theta,Phi);  cube T29m9 = getT(2,9,-9,g,Y9,R,Theta,Phi);
//  cube T09m8 = getT(0,9,-8,g,Y9,R,Theta,Phi);  cube T19m8 = getT(1,9,-8,g,Y9,R,Theta,Phi);  cube T29m8 = getT(2,9,-8,g,Y9,R,Theta,Phi);
//  cube T09m7 = getT(0,9,-7,g,Y9,R,Theta,Phi);  cube T19m7 = getT(1,9,-7,g,Y9,R,Theta,Phi);  cube T29m7 = getT(2,9,-7,g,Y9,R,Theta,Phi);
//  cube T09m6 = getT(0,9,-6,g,Y9,R,Theta,Phi);  cube T19m6 = getT(1,9,-6,g,Y9,R,Theta,Phi);  cube T29m6 = getT(2,9,-6,g,Y9,R,Theta,Phi);
//  cube T09m5 = getT(0,9,-5,g,Y9,R,Theta,Phi);  cube T19m5 = getT(1,9,-5,g,Y9,R,Theta,Phi);  cube T29m5 = getT(2,9,-5,g,Y9,R,Theta,Phi);
//  cube T09m4 = getT(0,9,-4,g,Y9,R,Theta,Phi);  cube T19m4 = getT(1,9,-4,g,Y9,R,Theta,Phi);  cube T29m4 = getT(2,9,-4,g,Y9,R,Theta,Phi);
//  cube T09m3 = getT(0,9,-3,g,Y9,R,Theta,Phi);  cube T19m3 = getT(1,9,-3,g,Y9,R,Theta,Phi);  cube T29m3 = getT(2,9,-3,g,Y9,R,Theta,Phi);
//  cube T09m2 = getT(0,9,-2,g,Y9,R,Theta,Phi);  cube T19m2 = getT(1,9,-2,g,Y9,R,Theta,Phi);  cube T29m2 = getT(2,9,-2,g,Y9,R,Theta,Phi);
//  cube T09m1 = getT(0,9,-1,g,Y9,R,Theta,Phi);  cube T19m1 = getT(1,9,-1,g,Y9,R,Theta,Phi);  cube T29m1 = getT(2,9,-1,g,Y9,R,Theta,Phi);
//  cube T090  = getT(0,9,+0,g,Y9,R,Theta,Phi);  cube T190  = getT(1,9,+0,g,Y9,R,Theta,Phi);  cube T290  = getT(2,9,+0,g,Y9,R,Theta,Phi);
//  cube T09p1 = getT(0,9,+1,g,Y9,R,Theta,Phi);  cube T19p1 = getT(1,9,+1,g,Y9,R,Theta,Phi);  cube T29p1 = getT(2,9,+1,g,Y9,R,Theta,Phi);
//  cube T09p2 = getT(0,9,+2,g,Y9,R,Theta,Phi);  cube T19p2 = getT(1,9,+2,g,Y9,R,Theta,Phi);  cube T29p2 = getT(2,9,+2,g,Y9,R,Theta,Phi);
//  cube T09p3 = getT(0,9,+3,g,Y9,R,Theta,Phi);  cube T19p3 = getT(1,9,+3,g,Y9,R,Theta,Phi);  cube T29p3 = getT(2,9,+3,g,Y9,R,Theta,Phi);
//  cube T09p4 = getT(0,9,+4,g,Y9,R,Theta,Phi);  cube T19p4 = getT(1,9,+4,g,Y9,R,Theta,Phi);  cube T29p4 = getT(2,9,+4,g,Y9,R,Theta,Phi);
//  cube T09p5 = getT(0,9,+5,g,Y9,R,Theta,Phi);  cube T19p5 = getT(1,9,+5,g,Y9,R,Theta,Phi);  cube T29p5 = getT(2,9,+5,g,Y9,R,Theta,Phi);
//  cube T09p6 = getT(0,9,+6,g,Y9,R,Theta,Phi);  cube T19p6 = getT(1,9,+6,g,Y9,R,Theta,Phi);  cube T29p6 = getT(2,9,+6,g,Y9,R,Theta,Phi);
//  cube T09p7 = getT(0,9,+7,g,Y9,R,Theta,Phi);  cube T19p7 = getT(1,9,+7,g,Y9,R,Theta,Phi);  cube T29p7 = getT(2,9,+7,g,Y9,R,Theta,Phi);
//  cube T09p8 = getT(0,9,+8,g,Y9,R,Theta,Phi);  cube T19p8 = getT(1,9,+8,g,Y9,R,Theta,Phi);  cube T29p8 = getT(2,9,+8,g,Y9,R,Theta,Phi);
//  cube T09p9 = getT(0,9,+9,g,Y9,R,Theta,Phi);  cube T19p9 = getT(1,9,+9,g,Y9,R,Theta,Phi);  cube T29p9 = getT(2,9,+9,g,Y9,R,Theta,Phi);


  //----------------------------------------------------------------------------------------------------------------
  // Part 3) get c_nlm by Integration -> Gauss quaduature used in 3D. Int Tnml(r,Theta,Phi) rho(r,theta,phi) dV
  //----------------------------------------------------------------------------------------------------------------

  cout << "FFF" << endl;

  double C000=rescaleInt*integrate3D(rho,T000,GLC,R,Theta,Phi);  double C100=rescaleInt*integrate3D(rho,T100,GLC,R,Theta,Phi);  double C200=rescaleInt*integrate3D(rho,T200,GLC,R,Theta,Phi);
  double C01m1=rescaleInt*integrate3D(rho,T01m1,GLC,R,Theta,Phi);double C11m1=rescaleInt*integrate3D(rho,T11m1,GLC,R,Theta,Phi);double C21m1=rescaleInt*integrate3D(rho,T21m1,GLC,R,Theta,Phi);
  double C010 =rescaleInt*integrate3D(rho,T010,GLC,R,Theta,Phi); double C110 =rescaleInt*integrate3D(rho,T110,GLC,R,Theta,Phi);double C210 =rescaleInt*integrate3D(rho,T210,GLC,R,Theta,Phi);
  double C01p1=rescaleInt*integrate3D(rho,T01p1,GLC,R,Theta,Phi);double C11p1=rescaleInt*integrate3D(rho,T11p1,GLC,R,Theta,Phi);double C21p1=rescaleInt*integrate3D(rho,T21p1,GLC,R,Theta,Phi);
  double C02m2=rescaleInt*integrate3D(rho,T02m2,GLC,R,Theta,Phi);double C12m2=rescaleInt*integrate3D(rho,T12m2,GLC,R,Theta,Phi);double C22m2=rescaleInt*integrate3D(rho,T22m2,GLC,R,Theta,Phi);
  double C02m1=rescaleInt*integrate3D(rho,T02m1,GLC,R,Theta,Phi);double C12m1=rescaleInt*integrate3D(rho,T12m1,GLC,R,Theta,Phi);double C22m1=rescaleInt*integrate3D(rho,T22m1,GLC,R,Theta,Phi);
  double C020 =rescaleInt*integrate3D(rho,T020,GLC,R,Theta,Phi); double C120 =rescaleInt*integrate3D(rho,T120,GLC,R,Theta,Phi);double C220 =rescaleInt*integrate3D(rho,T220,GLC,R,Theta,Phi);
  double C02p1=rescaleInt*integrate3D(rho,T02p1,GLC,R,Theta,Phi);double C12p1=rescaleInt*integrate3D(rho,T12p1,GLC,R,Theta,Phi);double C22p1=rescaleInt*integrate3D(rho,T22p1,GLC,R,Theta,Phi);
  double C02p2=rescaleInt*integrate3D(rho,T02p2,GLC,R,Theta,Phi);double C12p2=rescaleInt*integrate3D(rho,T12p2,GLC,R,Theta,Phi);double C22p2=rescaleInt*integrate3D(rho,T22p2,GLC,R,Theta,Phi);
  double C03m3=rescaleInt*integrate3D(rho,T03m3,GLC,R,Theta,Phi);double C13m3=rescaleInt*integrate3D(rho,T13m3,GLC,R,Theta,Phi);double C23m3=rescaleInt*integrate3D(rho,T23m3,GLC,R,Theta,Phi);
  double C03m2=rescaleInt*integrate3D(rho,T03m2,GLC,R,Theta,Phi);double C13m2=rescaleInt*integrate3D(rho,T13m2,GLC,R,Theta,Phi);double C23m2=rescaleInt*integrate3D(rho,T23m2,GLC,R,Theta,Phi);
  double C03m1=rescaleInt*integrate3D(rho,T03m1,GLC,R,Theta,Phi);double C13m1=rescaleInt*integrate3D(rho,T13m1,GLC,R,Theta,Phi);double C23m1=rescaleInt*integrate3D(rho,T23m1,GLC,R,Theta,Phi);
  double C030 =rescaleInt*integrate3D(rho,T030,GLC,R,Theta,Phi); double C130 =rescaleInt*integrate3D(rho,T130,GLC,R,Theta,Phi);double C230 =rescaleInt*integrate3D(rho,T230,GLC,R,Theta,Phi);
  double C03p1=rescaleInt*integrate3D(rho,T03p1,GLC,R,Theta,Phi);double C13p1=rescaleInt*integrate3D(rho,T13p1,GLC,R,Theta,Phi);double C23p1=rescaleInt*integrate3D(rho,T23p1,GLC,R,Theta,Phi);
  double C03p2=rescaleInt*integrate3D(rho,T03p2,GLC,R,Theta,Phi);double C13p2=rescaleInt*integrate3D(rho,T13p2,GLC,R,Theta,Phi);double C23p2=rescaleInt*integrate3D(rho,T23p2,GLC,R,Theta,Phi);
  double C03p3=rescaleInt*integrate3D(rho,T03p3,GLC,R,Theta,Phi);double C13p3=rescaleInt*integrate3D(rho,T13p3,GLC,R,Theta,Phi);double C23p3=rescaleInt*integrate3D(rho,T23p3,GLC,R,Theta,Phi);
  double C04m4=rescaleInt*integrate3D(rho,T04m4,GLC,R,Theta,Phi);double C14m4=rescaleInt*integrate3D(rho,T14m4,GLC,R,Theta,Phi);double C24m4=rescaleInt*integrate3D(rho,T24m4,GLC,R,Theta,Phi);
  double C04m3=rescaleInt*integrate3D(rho,T04m3,GLC,R,Theta,Phi);double C14m3=rescaleInt*integrate3D(rho,T14m3,GLC,R,Theta,Phi);double C24m3=rescaleInt*integrate3D(rho,T24m3,GLC,R,Theta,Phi);
  double C04m2=rescaleInt*integrate3D(rho,T04m2,GLC,R,Theta,Phi);double C14m2=rescaleInt*integrate3D(rho,T14m2,GLC,R,Theta,Phi);double C24m2=rescaleInt*integrate3D(rho,T24m2,GLC,R,Theta,Phi);
  double C04m1=rescaleInt*integrate3D(rho,T04m1,GLC,R,Theta,Phi);double C14m1=rescaleInt*integrate3D(rho,T14m1,GLC,R,Theta,Phi);double C24m1=rescaleInt*integrate3D(rho,T24m1,GLC,R,Theta,Phi);
  double C040 =rescaleInt*integrate3D(rho,T040,GLC,R,Theta,Phi); double C140 =rescaleInt*integrate3D(rho,T140,GLC,R,Theta,Phi);double C240 =rescaleInt*integrate3D(rho,T240,GLC,R,Theta,Phi);
  double C04p1=rescaleInt*integrate3D(rho,T04p1,GLC,R,Theta,Phi);double C14p1=rescaleInt*integrate3D(rho,T14p1,GLC,R,Theta,Phi);double C24p1=rescaleInt*integrate3D(rho,T24p1,GLC,R,Theta,Phi);
  double C04p2=rescaleInt*integrate3D(rho,T04p2,GLC,R,Theta,Phi);double C14p2=rescaleInt*integrate3D(rho,T14p2,GLC,R,Theta,Phi);double C24p2=rescaleInt*integrate3D(rho,T24p2,GLC,R,Theta,Phi);
  double C04p3=rescaleInt*integrate3D(rho,T04p3,GLC,R,Theta,Phi);double C14p3=rescaleInt*integrate3D(rho,T14p3,GLC,R,Theta,Phi);double C24p3=rescaleInt*integrate3D(rho,T24p3,GLC,R,Theta,Phi);
  double C04p4=rescaleInt*integrate3D(rho,T04p4,GLC,R,Theta,Phi);double C14p4=rescaleInt*integrate3D(rho,T14p4,GLC,R,Theta,Phi);double C24p4=rescaleInt*integrate3D(rho,T24p4,GLC,R,Theta,Phi);
  double C05m5=rescaleInt*integrate3D(rho,T05m5,GLC,R,Theta,Phi);double C15m5=rescaleInt*integrate3D(rho,T15m5,GLC,R,Theta,Phi);double C25m5=rescaleInt*integrate3D(rho,T25m5,GLC,R,Theta,Phi);
  double C05m4=rescaleInt*integrate3D(rho,T05m4,GLC,R,Theta,Phi);double C15m4=rescaleInt*integrate3D(rho,T15m4,GLC,R,Theta,Phi);double C25m4=rescaleInt*integrate3D(rho,T25m4,GLC,R,Theta,Phi);
  double C05m3=rescaleInt*integrate3D(rho,T05m3,GLC,R,Theta,Phi);double C15m3=rescaleInt*integrate3D(rho,T15m3,GLC,R,Theta,Phi);double C25m3=rescaleInt*integrate3D(rho,T25m3,GLC,R,Theta,Phi);
  double C05m2=rescaleInt*integrate3D(rho,T05m2,GLC,R,Theta,Phi);double C15m2=rescaleInt*integrate3D(rho,T15m2,GLC,R,Theta,Phi);double C25m2=rescaleInt*integrate3D(rho,T25m2,GLC,R,Theta,Phi);
  double C05m1=rescaleInt*integrate3D(rho,T05m1,GLC,R,Theta,Phi);double C15m1=rescaleInt*integrate3D(rho,T15m1,GLC,R,Theta,Phi);double C25m1=rescaleInt*integrate3D(rho,T25m1,GLC,R,Theta,Phi);
  double C050 =rescaleInt*integrate3D(rho,T050,GLC,R,Theta,Phi); double C150 =rescaleInt*integrate3D(rho,T150,GLC,R,Theta,Phi);double C250 =rescaleInt*integrate3D(rho,T250,GLC,R,Theta,Phi);
  double C05p1=rescaleInt*integrate3D(rho,T05p1,GLC,R,Theta,Phi);double C15p1=rescaleInt*integrate3D(rho,T15p1,GLC,R,Theta,Phi);double C25p1=rescaleInt*integrate3D(rho,T25p1,GLC,R,Theta,Phi);
  double C05p2=rescaleInt*integrate3D(rho,T05p2,GLC,R,Theta,Phi);double C15p2=rescaleInt*integrate3D(rho,T15p2,GLC,R,Theta,Phi);double C25p2=rescaleInt*integrate3D(rho,T25p2,GLC,R,Theta,Phi);
  double C05p3=rescaleInt*integrate3D(rho,T05p3,GLC,R,Theta,Phi);double C15p3=rescaleInt*integrate3D(rho,T15p3,GLC,R,Theta,Phi);double C25p3=rescaleInt*integrate3D(rho,T25p3,GLC,R,Theta,Phi);
  double C05p4=rescaleInt*integrate3D(rho,T05p4,GLC,R,Theta,Phi);double C15p4=rescaleInt*integrate3D(rho,T15p4,GLC,R,Theta,Phi);double C25p4=rescaleInt*integrate3D(rho,T25p4,GLC,R,Theta,Phi);
  double C05p5=rescaleInt*integrate3D(rho,T05p5,GLC,R,Theta,Phi);double C15p5=rescaleInt*integrate3D(rho,T15p5,GLC,R,Theta,Phi);double C25p5=rescaleInt*integrate3D(rho,T25p5,GLC,R,Theta,Phi);
  double C06m6=rescaleInt*integrate3D(rho,T06m6,GLC,R,Theta,Phi);double C16m6=rescaleInt*integrate3D(rho,T16m6,GLC,R,Theta,Phi);double C26m6=rescaleInt*integrate3D(rho,T26m6,GLC,R,Theta,Phi);
  double C06m5=rescaleInt*integrate3D(rho,T06m5,GLC,R,Theta,Phi);double C16m5=rescaleInt*integrate3D(rho,T16m5,GLC,R,Theta,Phi);double C26m5=rescaleInt*integrate3D(rho,T26m5,GLC,R,Theta,Phi);
  double C06m4=rescaleInt*integrate3D(rho,T06m4,GLC,R,Theta,Phi);double C16m4=rescaleInt*integrate3D(rho,T16m4,GLC,R,Theta,Phi);double C26m4=rescaleInt*integrate3D(rho,T26m4,GLC,R,Theta,Phi);
  double C06m3=rescaleInt*integrate3D(rho,T06m3,GLC,R,Theta,Phi);double C16m3=rescaleInt*integrate3D(rho,T16m3,GLC,R,Theta,Phi);double C26m3=rescaleInt*integrate3D(rho,T26m3,GLC,R,Theta,Phi);
  double C06m2=rescaleInt*integrate3D(rho,T06m2,GLC,R,Theta,Phi);double C16m2=rescaleInt*integrate3D(rho,T16m2,GLC,R,Theta,Phi);double C26m2=rescaleInt*integrate3D(rho,T26m2,GLC,R,Theta,Phi);
  double C06m1=rescaleInt*integrate3D(rho,T06m1,GLC,R,Theta,Phi);double C16m1=rescaleInt*integrate3D(rho,T16m1,GLC,R,Theta,Phi);double C26m1=rescaleInt*integrate3D(rho,T26m1,GLC,R,Theta,Phi);
  double C060 =rescaleInt*integrate3D(rho,T060,GLC,R,Theta,Phi); double C160 =rescaleInt*integrate3D(rho,T160,GLC,R,Theta,Phi);double C260 =rescaleInt*integrate3D(rho,T260,GLC,R,Theta,Phi);
  double C06p1=rescaleInt*integrate3D(rho,T06p1,GLC,R,Theta,Phi);double C16p1=rescaleInt*integrate3D(rho,T16p1,GLC,R,Theta,Phi);double C26p1=rescaleInt*integrate3D(rho,T26p1,GLC,R,Theta,Phi);
  double C06p2=rescaleInt*integrate3D(rho,T06p2,GLC,R,Theta,Phi);double C16p2=rescaleInt*integrate3D(rho,T16p2,GLC,R,Theta,Phi);double C26p2=rescaleInt*integrate3D(rho,T26p2,GLC,R,Theta,Phi);
  double C06p3=rescaleInt*integrate3D(rho,T06p3,GLC,R,Theta,Phi);double C16p3=rescaleInt*integrate3D(rho,T16p3,GLC,R,Theta,Phi);double C26p3=rescaleInt*integrate3D(rho,T26p3,GLC,R,Theta,Phi);
  double C06p4=rescaleInt*integrate3D(rho,T06p4,GLC,R,Theta,Phi);double C16p4=rescaleInt*integrate3D(rho,T16p4,GLC,R,Theta,Phi);double C26p4=rescaleInt*integrate3D(rho,T26p4,GLC,R,Theta,Phi);
  double C06p5=rescaleInt*integrate3D(rho,T06p5,GLC,R,Theta,Phi);double C16p5=rescaleInt*integrate3D(rho,T16p5,GLC,R,Theta,Phi);double C26p5=rescaleInt*integrate3D(rho,T26p5,GLC,R,Theta,Phi);
  double C06p6=rescaleInt*integrate3D(rho,T06p6,GLC,R,Theta,Phi);double C16p6=rescaleInt*integrate3D(rho,T16p6,GLC,R,Theta,Phi);double C26p6=rescaleInt*integrate3D(rho,T26p6,GLC,R,Theta,Phi);
  double C07m7=rescaleInt*integrate3D(rho,T07m7,GLC,R,Theta,Phi);double C17m7=rescaleInt*integrate3D(rho,T17m7,GLC,R,Theta,Phi);double C27m7=rescaleInt*integrate3D(rho,T27m7,GLC,R,Theta,Phi);
  double C07m6=rescaleInt*integrate3D(rho,T07m6,GLC,R,Theta,Phi);double C17m6=rescaleInt*integrate3D(rho,T17m6,GLC,R,Theta,Phi);double C27m6=rescaleInt*integrate3D(rho,T27m6,GLC,R,Theta,Phi);
  double C07m5=rescaleInt*integrate3D(rho,T07m5,GLC,R,Theta,Phi);double C17m5=rescaleInt*integrate3D(rho,T17m5,GLC,R,Theta,Phi);double C27m5=rescaleInt*integrate3D(rho,T27m5,GLC,R,Theta,Phi);
  double C07m4=rescaleInt*integrate3D(rho,T07m4,GLC,R,Theta,Phi);double C17m4=rescaleInt*integrate3D(rho,T17m4,GLC,R,Theta,Phi);double C27m4=rescaleInt*integrate3D(rho,T27m4,GLC,R,Theta,Phi);
  double C07m3=rescaleInt*integrate3D(rho,T07m3,GLC,R,Theta,Phi);double C17m3=rescaleInt*integrate3D(rho,T17m3,GLC,R,Theta,Phi);double C27m3=rescaleInt*integrate3D(rho,T27m3,GLC,R,Theta,Phi);
  double C07m2=rescaleInt*integrate3D(rho,T07m2,GLC,R,Theta,Phi);double C17m2=rescaleInt*integrate3D(rho,T17m2,GLC,R,Theta,Phi);double C27m2=rescaleInt*integrate3D(rho,T27m2,GLC,R,Theta,Phi);
  double C07m1=rescaleInt*integrate3D(rho,T07m1,GLC,R,Theta,Phi);double C17m1=rescaleInt*integrate3D(rho,T17m1,GLC,R,Theta,Phi);double C27m1=rescaleInt*integrate3D(rho,T27m1,GLC,R,Theta,Phi);
  double C070 =rescaleInt*integrate3D(rho,T070,GLC,R,Theta,Phi); double C170 =rescaleInt*integrate3D(rho,T170,GLC,R,Theta,Phi);double C270 =rescaleInt*integrate3D(rho,T270,GLC,R,Theta,Phi);
  double C07p1=rescaleInt*integrate3D(rho,T07p1,GLC,R,Theta,Phi);double C17p1=rescaleInt*integrate3D(rho,T17p1,GLC,R,Theta,Phi);double C27p1=rescaleInt*integrate3D(rho,T27p1,GLC,R,Theta,Phi);
  double C07p2=rescaleInt*integrate3D(rho,T07p2,GLC,R,Theta,Phi);double C17p2=rescaleInt*integrate3D(rho,T17p2,GLC,R,Theta,Phi);double C27p2=rescaleInt*integrate3D(rho,T27p2,GLC,R,Theta,Phi);
  double C07p3=rescaleInt*integrate3D(rho,T07p3,GLC,R,Theta,Phi);double C17p3=rescaleInt*integrate3D(rho,T17p3,GLC,R,Theta,Phi);double C27p3=rescaleInt*integrate3D(rho,T27p3,GLC,R,Theta,Phi);
  double C07p4=rescaleInt*integrate3D(rho,T07p4,GLC,R,Theta,Phi);double C17p4=rescaleInt*integrate3D(rho,T17p4,GLC,R,Theta,Phi);double C27p4=rescaleInt*integrate3D(rho,T27p4,GLC,R,Theta,Phi);
  double C07p5=rescaleInt*integrate3D(rho,T07p5,GLC,R,Theta,Phi);double C17p5=rescaleInt*integrate3D(rho,T17p5,GLC,R,Theta,Phi);double C27p5=rescaleInt*integrate3D(rho,T27p5,GLC,R,Theta,Phi);
  double C07p6=rescaleInt*integrate3D(rho,T07p6,GLC,R,Theta,Phi);double C17p6=rescaleInt*integrate3D(rho,T17p6,GLC,R,Theta,Phi);double C27p6=rescaleInt*integrate3D(rho,T27p6,GLC,R,Theta,Phi);
  double C07p7=rescaleInt*integrate3D(rho,T07p7,GLC,R,Theta,Phi);double C17p7=rescaleInt*integrate3D(rho,T17p7,GLC,R,Theta,Phi);double C27p7=rescaleInt*integrate3D(rho,T27p7,GLC,R,Theta,Phi);
//  double C08m8=rescaleInt*integrate3D(rho,T08m8,GLC,R,Theta,Phi); double C18m8=rescaleInt*integrate3D(rho,T18m8,GLC,R,Theta,Phi); double C28m8=rescaleInt*integrate3D(rho,T28m8,GLC,R,Theta,Phi);
//  double C08m7=rescaleInt*integrate3D(rho,T08m7,GLC,R,Theta,Phi); double C18m7=rescaleInt*integrate3D(rho,T18m7,GLC,R,Theta,Phi); double C28m7=rescaleInt*integrate3D(rho,T28m7,GLC,R,Theta,Phi);
//  double C08m6=rescaleInt*integrate3D(rho,T08m6,GLC,R,Theta,Phi); double C18m6=rescaleInt*integrate3D(rho,T18m6,GLC,R,Theta,Phi); double C28m6=rescaleInt*integrate3D(rho,T28m6,GLC,R,Theta,Phi);
//  double C08m5=rescaleInt*integrate3D(rho,T08m5,GLC,R,Theta,Phi); double C18m5=rescaleInt*integrate3D(rho,T18m5,GLC,R,Theta,Phi); double C28m5=rescaleInt*integrate3D(rho,T28m5,GLC,R,Theta,Phi);
//  double C08m4=rescaleInt*integrate3D(rho,T08m4,GLC,R,Theta,Phi); double C18m4=rescaleInt*integrate3D(rho,T18m4,GLC,R,Theta,Phi); double C28m4=rescaleInt*integrate3D(rho,T28m4,GLC,R,Theta,Phi);
//  double C08m3=rescaleInt*integrate3D(rho,T08m3,GLC,R,Theta,Phi); double C18m3=rescaleInt*integrate3D(rho,T18m3,GLC,R,Theta,Phi); double C28m3=rescaleInt*integrate3D(rho,T28m3,GLC,R,Theta,Phi);
//  double C08m2=rescaleInt*integrate3D(rho,T08m2,GLC,R,Theta,Phi); double C18m2=rescaleInt*integrate3D(rho,T18m2,GLC,R,Theta,Phi); double C28m2=rescaleInt*integrate3D(rho,T28m2,GLC,R,Theta,Phi);
//  double C08m1=rescaleInt*integrate3D(rho,T08m1,GLC,R,Theta,Phi); double C18m1=rescaleInt*integrate3D(rho,T18m1,GLC,R,Theta,Phi); double C28m1=rescaleInt*integrate3D(rho,T28m1,GLC,R,Theta,Phi);
//  double C080 =rescaleInt*integrate3D(rho,T080,GLC,R,Theta,Phi);  double C180 =rescaleInt*integrate3D(rho,T180,GLC,R,Theta,Phi); double C280 =rescaleInt*integrate3D(rho,T280,GLC,R,Theta,Phi);
//  double C08p1=rescaleInt*integrate3D(rho,T08p1,GLC,R,Theta,Phi); double C18p1=rescaleInt*integrate3D(rho,T18p1,GLC,R,Theta,Phi); double C28p1=rescaleInt*integrate3D(rho,T28p1,GLC,R,Theta,Phi);
//  double C08p2=rescaleInt*integrate3D(rho,T08p2,GLC,R,Theta,Phi); double C18p2=rescaleInt*integrate3D(rho,T18p2,GLC,R,Theta,Phi); double C28p2=rescaleInt*integrate3D(rho,T28p2,GLC,R,Theta,Phi);
//  double C08p3=rescaleInt*integrate3D(rho,T08p3,GLC,R,Theta,Phi); double C18p3=rescaleInt*integrate3D(rho,T18p3,GLC,R,Theta,Phi); double C28p3=rescaleInt*integrate3D(rho,T28p3,GLC,R,Theta,Phi);
//  double C08p4=rescaleInt*integrate3D(rho,T08p4,GLC,R,Theta,Phi); double C18p4=rescaleInt*integrate3D(rho,T18p4,GLC,R,Theta,Phi); double C28p4=rescaleInt*integrate3D(rho,T28p4,GLC,R,Theta,Phi);
//  double C08p5=rescaleInt*integrate3D(rho,T08p5,GLC,R,Theta,Phi); double C18p5=rescaleInt*integrate3D(rho,T18p5,GLC,R,Theta,Phi); double C28p5=rescaleInt*integrate3D(rho,T28p5,GLC,R,Theta,Phi);
//  double C08p6=rescaleInt*integrate3D(rho,T08p6,GLC,R,Theta,Phi); double C18p6=rescaleInt*integrate3D(rho,T18p6,GLC,R,Theta,Phi); double C28p6=rescaleInt*integrate3D(rho,T28p6,GLC,R,Theta,Phi);
//  double C08p7=rescaleInt*integrate3D(rho,T08p7,GLC,R,Theta,Phi); double C18p7=rescaleInt*integrate3D(rho,T18p7,GLC,R,Theta,Phi); double C28p7=rescaleInt*integrate3D(rho,T28p7,GLC,R,Theta,Phi);
//  double C08p8=rescaleInt*integrate3D(rho,T08p8,GLC,R,Theta,Phi); double C18p8=rescaleInt*integrate3D(rho,T18p8,GLC,R,Theta,Phi); double C28p8=rescaleInt*integrate3D(rho,T28p8,GLC,R,Theta,Phi);
//  double C09m9=rescaleInt*integrate3D(rho,T09m9,GLC,R,Theta,Phi); double C19m9=rescaleInt*integrate3D(rho,T19m9,GLC,R,Theta,Phi); double C29m9=rescaleInt*integrate3D(rho,T29m9,GLC,R,Theta,Phi);
//  double C09m8=rescaleInt*integrate3D(rho,T09m8,GLC,R,Theta,Phi); double C19m8=rescaleInt*integrate3D(rho,T19m8,GLC,R,Theta,Phi); double C29m8=rescaleInt*integrate3D(rho,T29m8,GLC,R,Theta,Phi);
//  double C09m7=rescaleInt*integrate3D(rho,T09m7,GLC,R,Theta,Phi); double C19m7=rescaleInt*integrate3D(rho,T19m7,GLC,R,Theta,Phi); double C29m7=rescaleInt*integrate3D(rho,T29m7,GLC,R,Theta,Phi);
//  double C09m6=rescaleInt*integrate3D(rho,T09m6,GLC,R,Theta,Phi); double C19m6=rescaleInt*integrate3D(rho,T19m6,GLC,R,Theta,Phi); double C29m6=rescaleInt*integrate3D(rho,T29m6,GLC,R,Theta,Phi);
//  double C09m5=rescaleInt*integrate3D(rho,T09m5,GLC,R,Theta,Phi); double C19m5=rescaleInt*integrate3D(rho,T19m5,GLC,R,Theta,Phi); double C29m5=rescaleInt*integrate3D(rho,T29m5,GLC,R,Theta,Phi);
//  double C09m4=rescaleInt*integrate3D(rho,T09m4,GLC,R,Theta,Phi); double C19m4=rescaleInt*integrate3D(rho,T19m4,GLC,R,Theta,Phi); double C29m4=rescaleInt*integrate3D(rho,T29m4,GLC,R,Theta,Phi);
//  double C09m3=rescaleInt*integrate3D(rho,T09m3,GLC,R,Theta,Phi); double C19m3=rescaleInt*integrate3D(rho,T19m3,GLC,R,Theta,Phi); double C29m3=rescaleInt*integrate3D(rho,T29m3,GLC,R,Theta,Phi);
//  double C09m2=rescaleInt*integrate3D(rho,T09m2,GLC,R,Theta,Phi); double C19m2=rescaleInt*integrate3D(rho,T19m2,GLC,R,Theta,Phi); double C29m2=rescaleInt*integrate3D(rho,T29m2,GLC,R,Theta,Phi);
//  double C09m1=rescaleInt*integrate3D(rho,T09m1,GLC,R,Theta,Phi); double C19m1=rescaleInt*integrate3D(rho,T19m1,GLC,R,Theta,Phi); double C29m1=rescaleInt*integrate3D(rho,T29m1,GLC,R,Theta,Phi);
//  double C090 =rescaleInt*integrate3D(rho,T090,GLC,R,Theta,Phi);  double C190 =rescaleInt*integrate3D(rho,T190,GLC,R,Theta,Phi); double C290 =rescaleInt*integrate3D(rho,T290,GLC,R,Theta,Phi);
//  double C09p1=rescaleInt*integrate3D(rho,T09p1,GLC,R,Theta,Phi); double C19p1=rescaleInt*integrate3D(rho,T19p1,GLC,R,Theta,Phi); double C29p1=rescaleInt*integrate3D(rho,T29p1,GLC,R,Theta,Phi);
//  double C09p2=rescaleInt*integrate3D(rho,T09p2,GLC,R,Theta,Phi); double C19p2=rescaleInt*integrate3D(rho,T19p2,GLC,R,Theta,Phi); double C29p2=rescaleInt*integrate3D(rho,T29p2,GLC,R,Theta,Phi);
//  double C09p3=rescaleInt*integrate3D(rho,T09p3,GLC,R,Theta,Phi); double C19p3=rescaleInt*integrate3D(rho,T19p3,GLC,R,Theta,Phi); double C29p3=rescaleInt*integrate3D(rho,T29p3,GLC,R,Theta,Phi);
//  double C09p4=rescaleInt*integrate3D(rho,T09p4,GLC,R,Theta,Phi); double C19p4=rescaleInt*integrate3D(rho,T19p4,GLC,R,Theta,Phi); double C29p4=rescaleInt*integrate3D(rho,T29p4,GLC,R,Theta,Phi);
//  double C09p5=rescaleInt*integrate3D(rho,T09p5,GLC,R,Theta,Phi); double C19p5=rescaleInt*integrate3D(rho,T19p5,GLC,R,Theta,Phi); double C29p5=rescaleInt*integrate3D(rho,T29p5,GLC,R,Theta,Phi);
//  double C09p6=rescaleInt*integrate3D(rho,T09p6,GLC,R,Theta,Phi); double C19p6=rescaleInt*integrate3D(rho,T19p6,GLC,R,Theta,Phi); double C29p6=rescaleInt*integrate3D(rho,T29p6,GLC,R,Theta,Phi);
//  double C09p7=rescaleInt*integrate3D(rho,T09p7,GLC,R,Theta,Phi); double C19p7=rescaleInt*integrate3D(rho,T19p7,GLC,R,Theta,Phi); double C29p7=rescaleInt*integrate3D(rho,T29p7,GLC,R,Theta,Phi);
//  double C09p8=rescaleInt*integrate3D(rho,T09p8,GLC,R,Theta,Phi); double C19p8=rescaleInt*integrate3D(rho,T19p8,GLC,R,Theta,Phi); double C29p8=rescaleInt*integrate3D(rho,T29p8,GLC,R,Theta,Phi);
//  double C09p9=rescaleInt*integrate3D(rho,T09p9,GLC,R,Theta,Phi); double C19p9=rescaleInt*integrate3D(rho,T19p9,GLC,R,Theta,Phi); double C29p9=rescaleInt*integrate3D(rho,T29p9,GLC,R,Theta,Phi);


 // Just printing out for Debug. 
  cout << C000 << " " <<  C100 << " " <<  C200 << endl;
  cout << C01m1 << " " <<  C11m1 << " " <<  C21m1 << endl;
  cout << C010  << " " <<  C110  << " " <<  C210  << endl;
  cout << C01p1 << " " <<  C11p1 << " " <<  C21p1 << endl;
  cout << C02m2 << " " <<  C12m2 << " " <<  C22m2 << endl;
  cout << C02m1 << " " <<  C12m1 << " " <<  C22m1 << endl;
  cout << C020  << " " <<  C120  << " " <<  C220  << endl;
  cout << C02p1 << " " <<  C12p1 << " " <<  C22p1 << endl;
  cout << C02p2 << " " <<  C12p2 << " " <<  C22p2 << endl;
  cout << C03m3 << " " <<  C13m3 << " " <<  C23m3 << endl;
  cout << C03m2 << " " <<  C13m2 << " " <<  C23m2 << endl;
  cout << C03m1 << " " <<  C13m1 << " " <<  C23m1 << endl;
  cout << C030  << " " <<  C130  << " " <<  C230  << endl;
  cout << C03p1 << " " <<  C13p1 << " " <<  C23p1 << endl;
  cout << C03p2 << " " <<  C13p2 << " " <<  C23p2 << endl;
  cout << C03p3 << " " <<  C13p3 << " " <<  C23p3 << endl;
  cout << C04m4 << " " <<  C14m4 << " " <<  C24m4 << endl;
  cout << C04m3 << " " <<  C14m3 << " " <<  C24m3 << endl;
  cout << C04m2 << " " <<  C14m2 << " " <<  C24m2 << endl;
  cout << C04m1 << " " <<  C14m1 << " " <<  C24m1 << endl;
  cout << C040  << " " <<  C140  << " " <<  C240  << endl;
  cout << C04p1 << " " <<  C14p1 << " " <<  C24p1 << endl;
  cout << C04p2 << " " <<  C14p2 << " " <<  C24p2 << endl;
  cout << C04p3 << " " <<  C14p3 << " " <<  C24p3 << endl;
  cout << C04p4 << " " <<  C14p4 << " " <<  C24p4 << endl;
  cout << C05m5 << " " <<  C15m5 << " " <<  C25m5 << endl;
  cout << C05m4 << " " <<  C15m4 << " " <<  C25m4 << endl;
  cout << C05m3 << " " <<  C15m3 << " " <<  C25m3 << endl;
  cout << C05m2 << " " <<  C15m2 << " " <<  C25m2 << endl;
  cout << C05m1 << " " <<  C15m1 << " " <<  C25m1 << endl;
  cout << C050  << " " <<  C150  << " " <<  C250  << endl;
  cout << C05p1 << " " <<  C15p1 << " " <<  C25p1 << endl;
  cout << C05p2 << " " <<  C15p2 << " " <<  C25p2 << endl;
  cout << C05p3 << " " <<  C15p3 << " " <<  C25p3 << endl;
  cout << C05p4 << " " <<  C15p4 << " " <<  C25p4 << endl;
  cout << C05p5 << " " <<  C15p5 << " " <<  C25p5 << endl;
  cout << C06m6 << " " <<  C16m6 << " " <<  C26m6 << endl;
  cout << C06m5 << " " <<  C16m5 << " " <<  C26m5 << endl;
  cout << C06m4 << " " <<  C16m4 << " " <<  C26m4 << endl;
  cout << C06m3 << " " <<  C16m3 << " " <<  C26m3 << endl;
  cout << C06m2 << " " <<  C16m2 << " " <<  C26m2 << endl;
  cout << C06m1 << " " <<  C16m1 << " " <<  C26m1 << endl;
  cout << C060  << " " <<  C160  << " " <<  C260  << endl;
  cout << C06p1 << " " <<  C16p1 << " " <<  C26p1 << endl;
  cout << C06p2 << " " <<  C16p2 << " " <<  C26p2 << endl;
  cout << C06p3 << " " <<  C16p3 << " " <<  C26p3 << endl;
  cout << C06p4 << " " <<  C16p4 << " " <<  C26p4 << endl;
  cout << C06p5 << " " <<  C16p5 << " " <<  C26p5 << endl;
  cout << C06p6 << " " <<  C16p6 << " " <<  C26p6 << endl;
  cout << C07m7 << " " <<  C17m7 << " " <<  C27m7 << endl;
  cout << C07m6 << " " <<  C17m6 << " " <<  C27m6 << endl;
  cout << C07m5 << " " <<  C17m5 << " " <<  C27m5 << endl;
  cout << C07m4 << " " <<  C17m4 << " " <<  C27m4 << endl;
  cout << C07m3 << " " <<  C17m3 << " " <<  C27m3 << endl;
  cout << C07m2 << " " <<  C17m2 << " " <<  C27m2 << endl;
  cout << C07m1 << " " <<  C17m1 << " " <<  C27m1 << endl;
  cout << C070  << " " <<  C170  << " " <<  C270  << endl;
  cout << C07p1 << " " <<  C17p1 << " " <<  C27p1 << endl;
  cout << C07p2 << " " <<  C17p2 << " " <<  C27p2 << endl;
  cout << C07p3 << " " <<  C17p3 << " " <<  C27p3 << endl;
  cout << C07p4 << " " <<  C17p4 << " " <<  C27p4 << endl;
  cout << C07p5 << " " <<  C17p5 << " " <<  C27p5 << endl;
  cout << C07p6 << " " <<  C17p6 << " " <<  C27p6 << endl;
  cout << C07p7 << " " <<  C17p7 << " " <<  C27p7 << endl;
//  cout << C08m8 << " " <<  C18m8 << " " <<  C28m8 << endl;
//  cout << C08m7 << " " <<  C18m7 << " " <<  C28m7 << endl;
//  cout << C08m6 << " " <<  C18m6 << " " <<  C28m6 << endl;
//  cout << C08m5 << " " <<  C18m5 << " " <<  C28m5 << endl;
//  cout << C08m4 << " " <<  C18m4 << " " <<  C28m4 << endl;
//  cout << C08m3 << " " <<  C18m3 << " " <<  C28m3 << endl;
//  cout << C08m2 << " " <<  C18m2 << " " <<  C28m2 << endl;
//  cout << C08m1 << " " <<  C18m1 << " " <<  C28m1 << endl;
//  cout << C080  << " " <<  C180  << " " <<  C280  << endl;
//  cout << C08p1 << " " <<  C18p1 << " " <<  C28p1 << endl;
//  cout << C08p2 << " " <<  C18p2 << " " <<  C28p2 << endl;
//  cout << C08p3 << " " <<  C18p3 << " " <<  C28p3 << endl;
//  cout << C08p4 << " " <<  C18p4 << " " <<  C28p4 << endl;
//  cout << C08p5 << " " <<  C18p5 << " " <<  C28p5 << endl;
//  cout << C08p6 << " " <<  C18p6 << " " <<  C28p6 << endl;
//  cout << C08p7 << " " <<  C18p7 << " " <<  C28p7 << endl;
//  cout << C08p8 << " " <<  C18p8 << " " <<  C28p8 << endl;
//  cout << C09m9 << " " <<  C19m9 << " " <<  C29m9 << endl;
//  cout << C09m8 << " " <<  C19m8 << " " <<  C29m8 << endl;
//  cout << C09m7 << " " <<  C19m7 << " " <<  C29m7 << endl;
//  cout << C09m6 << " " <<  C19m6 << " " <<  C29m6 << endl;
//  cout << C09m5 << " " <<  C19m5 << " " <<  C29m5 << endl;
//  cout << C09m4 << " " <<  C19m4 << " " <<  C29m4 << endl;
//  cout << C09m3 << " " <<  C19m3 << " " <<  C29m3 << endl;
//  cout << C09m2 << " " <<  C19m2 << " " <<  C29m2 << endl;
//  cout << C09m1 << " " <<  C19m1 << " " <<  C29m1 << endl;
//  cout << C090  << " " <<  C190  << " " <<  C290  << endl;
//  cout << C09p1 << " " <<  C19p1 << " " <<  C29p1 << endl;
//  cout << C09p2 << " " <<  C19p2 << " " <<  C29p2 << endl;
//  cout << C09p3 << " " <<  C19p3 << " " <<  C29p3 << endl;
//  cout << C09p4 << " " <<  C19p4 << " " <<  C29p4 << endl;
//  cout << C09p5 << " " <<  C19p5 << " " <<  C29p5 << endl;
//  cout << C09p6 << " " <<  C19p6 << " " <<  C29p6 << endl;
//  cout << C09p7 << " " <<  C19p7 << " " <<  C29p7 << endl;
//  cout << C09p8 << " " <<  C19p8 << " " <<  C29p8 << endl;
//  cout << C09p9 << " " <<  C19p9 << " " <<  C29p9 << endl;


// cube Tnl = getT(n,l,m,g,Yl,GL);
//      cout << r << " " << t <<  " " << p << " " << endl;  
  
//  *create tensor T[n][r][t][p] = r**2*g(r)*Y(t,p)*sin(t);




return 0;
}
