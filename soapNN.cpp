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
  // 2) Constructing Basis functions -> g_n(r)[APB eq.25], Y_lm(The,Phi)[Tesseral Harmonics], where The=Theta      //
  // 3) Preperation for integration. -> T[n][l][m][r][The][Phi] = r^2 sin(The) g_n(r) Y_lm(The,Phi).               //
  //                                     ( Tensor that is independent of the inputs.)                              //
  // 4) Get coeffs c_nlm[APD eq.24] = Integrate Rho(r,The,Phi) T dV, where Rho[SD eq.14] is the Gaussian smeared   //
  //                                    xyz atomic positions.                                                      //
  // 5) Get power spectrums by P_b1b2lab = sum_m (c_b1lma  c_b2lmb) [SD eq.17]                                     //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //----------------------------------------------------------------------------------------------------------------
  // Part 0) Setting Parameters
  //----------------------------------------------------------------------------------------------------------------
 // int lMax = 9;
  
  double pi = 3.14159265358979324;
  double halfPi = 3.14159265358979324*0.5;


  double rcut = 20.0;
  double rsc = pi*pi*0.5*0.5*rcut; // rescaleing the integration for gauss-legendre quaduature.

  double sig = 1;

  double ao = 2.0;
  double z = 1.0;
  double norm = pow(sqrt(z/ao),3);

  
  //----------------------------------------------------------------------------------------------------------------
  // Part 1) Retrieving Data -> W, Gauss-Legendre, XYZ-Smeared
  //----------------------------------------------------------------------------------------------------------------
  
  mat W; // [APB eq.25]
  W.load("WMatData/W3.csv");

  mat GL; // [http://keisan.casio.com/exec/system/1329114617 (June 5th 2017)] , produced by Octave. W(:,0) -> GL coord. pos. W(:,1) -> GL weights.
  GL.load("parameters100.txt");

  vec R = rcut*0.5*GL.col(0) + rcut*0.5 ; // rescaled R for gauss-legendre quaduature
  vec The = pi*GL.col(0)*0.5 + pi*0.5;  // rescaled The for gauss-legendre quaduature
  vec Phi = pi*GL.col(0) + pi;            // rescaled Phi for gauss-legendre quaduature

  cube X ;
  cube Y ;
  cube Z ;
  X.load("X.bi");
  Y.load("Y.bi");
  Z.load("Z.bi");

//  X = getSphericalToCartCubeX( R, The, Phi);
//  Y = getSphericalToCartCubeY( R, The, Phi);
//  Z = getSphericalToCartCubeZ( R, The, Phi);
//  X.save("X.bi");
//  Y.save("Y.bi");
//  Z.save("Z.bi");


  mat coord = getPos(argv[1]);
  coord = posAve(coord); 
  coord = rotate3d(coord,1,1,1);
  string* type = getType(argv[1]);

  cube rho = getGaussDistr(coord,R, The, Phi, X, Y, Z, sig);
//  rho.print("rho");

//  mat printRho(100,100);
//  for(int i=0; i < 100; i++) 
//    for(int j=0; j < 100; j++)    printRho(i,j) = rho.at(15,i,j); // sanity check for Gaussian distribution of atoms.

  vec lastAtom = coord.row(coord.n_rows - 1).t();

  cube GLC(GL.n_rows,GL.n_rows,GL.n_rows);
  GLC.load("GLC.bi");
//  for(int i=0; i < GL.n_rows; i++){ 
//    for(int j=0; j < GL.n_rows; j++){ 
//      for(int k=0; k < GL.n_rows; k++){ 

//        GLC.at(i,j,k) = GL.at(i,1)*GL.at(j,1)*GL.at(k,1); // Setting up the GL weights.

//      }
//    }
//  }

  cout << "Part 1: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 2) Constructing Basis Functions -> g_n(r), N_a, phi_a(r),  
  //----------------------------------------------------------------------------------------------------------------
  
//  vec N = zeros<vec>(3); //[APB eq.25]

//  for(int i=0; i < N.n_rows; i++){ 
//    N(i) = sqrt(pow(rcut, 2*(i + 1) + 5)/(2*(i + 1) + 5)); // DO I NEED TO RESCAEL rcut???
//   }

//  mat phi = zeros<mat>(N.n_elem,GL.n_rows); //[APB eq.25]
//  for(int r=0; r < R.n_rows; r++){ 
//    for(int a=0; a < N.n_rows; a++){ 
////      phi(a,r) = pow(rcut - GL(r,0), 2*(a + 3))/N(a); // not rescaled
//      phi(a,r) = pow(rcut - R(r), 2*(a + 3))/N(a); // rescaled
//    }
//  }


//  mat g = zeros<mat>(N.n_rows,GL.n_rows); // Radial Basis Functions [APB eq.25]. g(*,:) -> n's. g(:,*) -> r's of GL coord. pos.
//  for(int r=0; r < GL.n_rows; r++){ 
//    for(int n=0; n < N.n_rows; n++){ 
//      for(int a=0; a < W.n_rows; a++){ 
//        g.at(n,r) = g.at(n,r) + W.at(n,a)*phi.at(a,r); // DO I NEED TO RESCALE W???
//      }
//    }
//  }
  mat gn(R.n_rows,4);
  gn.col(0)= hydrogenRDF(1, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
  gn.col(1)= hydrogenRDF(2, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
  gn.col(2)= hydrogenRDF(3, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
  gn.col(3)= hydrogenRDF(4, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)

  mat    g = gn.t();

//N.save("N.dat",raw_ascii);
//printRho.save("Rho.dat",raw_ascii);
g.save("gn.dat",raw_ascii);
//phi.save("phi.dat",raw_ascii);
R.save("R.dat",raw_ascii);
The.save("The.dat",raw_ascii);
Phi.save("Phi.dat",raw_ascii);


  mat Y0 = zeros<mat>(GL.n_rows, GL.n_rows);// Tesseral Spherical Harmonics at GL coord. pos. first lMax is l, second lMax is m but patted with 0's. 
  for(int t=0; t < GL.n_rows; t++){ 
    for(int p=0; p < GL.n_rows; p++){ 
  //    Y0(t,p) = tesseral_spherical_harm(0,0,GL(t,0),GL(p,0)); // not rescaled.
       Y0.at(t,p) = tesseral_spherical_harm(0,0,The.at(t),Phi.at(p)); // rescaled
      }
    }

  cube Y1 = getY(1,The, Phi);
  cube Y2 = getY(2,The, Phi);
  cube Y3 = getY(3,The, Phi);
  cube Y4 = getY(4,The, Phi);
  cube Y5 = getY(5,The, Phi);
  cube Y6 = getY(6,The, Phi);
  cube Y7 = getY(7,The, Phi);
  cube Y8 = getY(8,The, Phi);
  cube Y9 = getY(9,The, Phi);
   
  cout << "Part 2: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 3) Preparing Integrand -> Cube Tnlm(r,The,Phi) Due to memory problem, the T's and C's are split into many
  // variables;
  //----------------------------------------------------------------------------------------------------------------

  cube T000 = getTMat(0,0,0,g,Y0,R,The,Phi); cube T100 = getTMat(1,0,0,g,Y0,R,The,Phi);  cube T200 = getTMat(2,0,0,g,Y0,R,The,Phi);  cube T300 = getTMat(3,0,0,g,Y0,R,The,Phi);
  cube T01m1 = getT(0,1,-1,g,Y1,R,The,Phi);  cube T11m1 = getT(1,1,-1,g,Y1,R,The,Phi);  cube T21m1 = getT(2,1,-1,g,Y1,R,The,Phi);  cube T31m1 = getT(3,1,-1,g,Y1,R,The,Phi);
  cube T010  = getT(0,1,+0,g,Y1,R,The,Phi);  cube T110  = getT(1,1,+0,g,Y1,R,The,Phi);  cube T210  = getT(2,1,+0,g,Y1,R,The,Phi);  cube T310  = getT(3,1,+0,g,Y1,R,The,Phi);
  cube T01p1 = getT(0,1,+1,g,Y1,R,The,Phi);  cube T11p1 = getT(1,1,+1,g,Y1,R,The,Phi);  cube T21p1 = getT(2,1,+1,g,Y1,R,The,Phi);  cube T31p1 = getT(3,1,+1,g,Y1,R,The,Phi);
  cube T02m2 = getT(0,2,-2,g,Y2,R,The,Phi);  cube T12m2 = getT(1,2,-2,g,Y2,R,The,Phi);  cube T22m2 = getT(2,2,-2,g,Y2,R,The,Phi);  cube T32m2 = getT(3,2,-2,g,Y2,R,The,Phi);
  cube T02m1 = getT(0,2,-1,g,Y2,R,The,Phi);  cube T12m1 = getT(1,2,-1,g,Y2,R,The,Phi);  cube T22m1 = getT(2,2,-1,g,Y2,R,The,Phi);  cube T32m1 = getT(3,2,-1,g,Y2,R,The,Phi);
  cube T020  = getT(0,2,+0,g,Y2,R,The,Phi);  cube T120  = getT(1,2,+0,g,Y2,R,The,Phi);  cube T220  = getT(2,2,+0,g,Y2,R,The,Phi);  cube T320  = getT(3,2,+0,g,Y2,R,The,Phi);
  cube T02p1 = getT(0,2,+1,g,Y2,R,The,Phi);  cube T12p1 = getT(1,2,+1,g,Y2,R,The,Phi);  cube T22p1 = getT(2,2,+1,g,Y2,R,The,Phi);  cube T32p1 = getT(3,2,+1,g,Y2,R,The,Phi);
  cube T02p2 = getT(0,2,+2,g,Y2,R,The,Phi);  cube T12p2 = getT(1,2,+2,g,Y2,R,The,Phi);  cube T22p2 = getT(2,2,+2,g,Y2,R,The,Phi);  cube T32p2 = getT(3,2,+2,g,Y2,R,The,Phi);
  cube T03m3 = getT(0,3,-3,g,Y3,R,The,Phi);  cube T13m3 = getT(1,3,-3,g,Y3,R,The,Phi);  cube T23m3 = getT(2,3,-3,g,Y3,R,The,Phi);  cube T33m3 = getT(3,3,-3,g,Y3,R,The,Phi);
  cube T03m2 = getT(0,3,-2,g,Y3,R,The,Phi);  cube T13m2 = getT(1,3,-2,g,Y3,R,The,Phi);  cube T23m2 = getT(2,3,-2,g,Y3,R,The,Phi);  cube T33m2 = getT(3,3,-2,g,Y3,R,The,Phi);
  cube T03m1 = getT(0,3,-1,g,Y3,R,The,Phi);  cube T13m1 = getT(1,3,-1,g,Y3,R,The,Phi);  cube T23m1 = getT(2,3,-1,g,Y3,R,The,Phi);  cube T33m1 = getT(3,3,-1,g,Y3,R,The,Phi);
  cube T030  = getT(0,3,+0,g,Y3,R,The,Phi);  cube T130  = getT(1,3,+0,g,Y3,R,The,Phi);  cube T230  = getT(2,3,+0,g,Y3,R,The,Phi);  cube T330  = getT(3,3,+0,g,Y3,R,The,Phi);
  cube T03p1 = getT(0,3,+1,g,Y3,R,The,Phi);  cube T13p1 = getT(1,3,+1,g,Y3,R,The,Phi);  cube T23p1 = getT(2,3,+1,g,Y3,R,The,Phi);  cube T33p1 = getT(3,3,+1,g,Y3,R,The,Phi);
  cube T03p2 = getT(0,3,+2,g,Y3,R,The,Phi);  cube T13p2 = getT(1,3,+2,g,Y3,R,The,Phi);  cube T23p2 = getT(2,3,+2,g,Y3,R,The,Phi);  cube T33p2 = getT(3,3,+2,g,Y3,R,The,Phi);
  cube T03p3 = getT(0,3,+3,g,Y3,R,The,Phi);  cube T13p3 = getT(1,3,+3,g,Y3,R,The,Phi);  cube T23p3 = getT(2,3,+3,g,Y3,R,The,Phi);  cube T33p3 = getT(3,3,+3,g,Y3,R,The,Phi);
  cube T04m4 = getT(0,4,-4,g,Y4,R,The,Phi);  cube T14m4 = getT(1,4,-4,g,Y4,R,The,Phi);  cube T24m4 = getT(2,4,-4,g,Y4,R,The,Phi);  cube T34m4 = getT(3,4,-4,g,Y4,R,The,Phi);
  cube T04m3 = getT(0,4,-3,g,Y4,R,The,Phi);  cube T14m3 = getT(1,4,-3,g,Y4,R,The,Phi);  cube T24m3 = getT(2,4,-3,g,Y4,R,The,Phi);  cube T34m3 = getT(3,4,-3,g,Y4,R,The,Phi);
  cube T04m2 = getT(0,4,-2,g,Y4,R,The,Phi);  cube T14m2 = getT(1,4,-2,g,Y4,R,The,Phi);  cube T24m2 = getT(2,4,-2,g,Y4,R,The,Phi);  cube T34m2 = getT(3,4,-2,g,Y4,R,The,Phi);
  cube T04m1 = getT(0,4,-1,g,Y4,R,The,Phi);  cube T14m1 = getT(1,4,-1,g,Y4,R,The,Phi);  cube T24m1 = getT(2,4,-1,g,Y4,R,The,Phi);  cube T34m1 = getT(3,4,-1,g,Y4,R,The,Phi);
  cube T040  = getT(0,4,+0,g,Y4,R,The,Phi);  cube T140  = getT(1,4,+0,g,Y4,R,The,Phi);  cube T240  = getT(2,4,+0,g,Y4,R,The,Phi);  cube T340  = getT(3,4,+0,g,Y4,R,The,Phi);
  cube T04p1 = getT(0,4,+1,g,Y4,R,The,Phi);  cube T14p1 = getT(1,4,+1,g,Y4,R,The,Phi);  cube T24p1 = getT(2,4,+1,g,Y4,R,The,Phi);  cube T34p1 = getT(3,4,+1,g,Y4,R,The,Phi);
  cube T04p2 = getT(0,4,+2,g,Y4,R,The,Phi);  cube T14p2 = getT(1,4,+2,g,Y4,R,The,Phi);  cube T24p2 = getT(2,4,+2,g,Y4,R,The,Phi);  cube T34p2 = getT(3,4,+2,g,Y4,R,The,Phi);
  cube T04p3 = getT(0,4,+3,g,Y4,R,The,Phi);  cube T14p3 = getT(1,4,+3,g,Y4,R,The,Phi);  cube T24p3 = getT(2,4,+3,g,Y4,R,The,Phi);  cube T34p3 = getT(3,4,+3,g,Y4,R,The,Phi);
  cube T04p4 = getT(0,4,+4,g,Y4,R,The,Phi);  cube T14p4 = getT(1,4,+4,g,Y4,R,The,Phi);  cube T24p4 = getT(2,4,+4,g,Y4,R,The,Phi);  cube T34p4 = getT(3,4,+4,g,Y4,R,The,Phi);
  cube T05m5 = getT(0,5,-5,g,Y5,R,The,Phi);  cube T15m5 = getT(1,5,-5,g,Y5,R,The,Phi);  cube T25m5 = getT(2,5,-5,g,Y5,R,The,Phi);  cube T35m5 = getT(3,5,-5,g,Y5,R,The,Phi);
  cube T05m4 = getT(0,5,-4,g,Y5,R,The,Phi);  cube T15m4 = getT(1,5,-4,g,Y5,R,The,Phi);  cube T25m4 = getT(2,5,-4,g,Y5,R,The,Phi);  cube T35m4 = getT(3,5,-4,g,Y5,R,The,Phi);
  cube T05m3 = getT(0,5,-3,g,Y5,R,The,Phi);  cube T15m3 = getT(1,5,-3,g,Y5,R,The,Phi);  cube T25m3 = getT(2,5,-3,g,Y5,R,The,Phi);  cube T35m3 = getT(3,5,-3,g,Y5,R,The,Phi);
  cube T05m2 = getT(0,5,-2,g,Y5,R,The,Phi);  cube T15m2 = getT(1,5,-2,g,Y5,R,The,Phi);  cube T25m2 = getT(2,5,-2,g,Y5,R,The,Phi);  cube T35m2 = getT(3,5,-2,g,Y5,R,The,Phi);
  cube T05m1 = getT(0,5,-1,g,Y5,R,The,Phi);  cube T15m1 = getT(1,5,-1,g,Y5,R,The,Phi);  cube T25m1 = getT(2,5,-1,g,Y5,R,The,Phi);  cube T35m1 = getT(3,5,-1,g,Y5,R,The,Phi);
  cube T050  = getT(0,5,+0,g,Y5,R,The,Phi);  cube T150  = getT(1,5,+0,g,Y5,R,The,Phi);  cube T250  = getT(2,5,+0,g,Y5,R,The,Phi);  cube T350  = getT(3,5,+0,g,Y5,R,The,Phi);
  cube T05p1 = getT(0,5,+1,g,Y5,R,The,Phi);  cube T15p1 = getT(1,5,+1,g,Y5,R,The,Phi);  cube T25p1 = getT(2,5,+1,g,Y5,R,The,Phi);  cube T35p1 = getT(3,5,+1,g,Y5,R,The,Phi);
  cube T05p2 = getT(0,5,+2,g,Y5,R,The,Phi);  cube T15p2 = getT(1,5,+2,g,Y5,R,The,Phi);  cube T25p2 = getT(2,5,+2,g,Y5,R,The,Phi);  cube T35p2 = getT(3,5,+2,g,Y5,R,The,Phi);
  cube T05p3 = getT(0,5,+3,g,Y5,R,The,Phi);  cube T15p3 = getT(1,5,+3,g,Y5,R,The,Phi);  cube T25p3 = getT(2,5,+3,g,Y5,R,The,Phi);  cube T35p3 = getT(3,5,+3,g,Y5,R,The,Phi);
  cube T05p4 = getT(0,5,+4,g,Y5,R,The,Phi);  cube T15p4 = getT(1,5,+4,g,Y5,R,The,Phi);  cube T25p4 = getT(2,5,+4,g,Y5,R,The,Phi);  cube T35p4 = getT(3,5,+4,g,Y5,R,The,Phi);
  cube T05p5 = getT(0,5,+5,g,Y5,R,The,Phi);  cube T15p5 = getT(1,5,+5,g,Y5,R,The,Phi);  cube T25p5 = getT(2,5,+5,g,Y5,R,The,Phi);  cube T35p5 = getT(3,5,+5,g,Y5,R,The,Phi);
  cube T06m6 = getT(0,6,-6,g,Y6,R,The,Phi);  cube T16m6 = getT(1,6,-6,g,Y6,R,The,Phi);  cube T26m6 = getT(2,6,-6,g,Y6,R,The,Phi);  cube T36m6 = getT(3,6,-6,g,Y6,R,The,Phi);
  cube T06m5 = getT(0,6,-5,g,Y6,R,The,Phi);  cube T16m5 = getT(1,6,-5,g,Y6,R,The,Phi);  cube T26m5 = getT(2,6,-5,g,Y6,R,The,Phi);  cube T36m5 = getT(3,6,-5,g,Y6,R,The,Phi);
  cube T06m4 = getT(0,6,-4,g,Y6,R,The,Phi);  cube T16m4 = getT(1,6,-4,g,Y6,R,The,Phi);  cube T26m4 = getT(2,6,-4,g,Y6,R,The,Phi);  cube T36m4 = getT(3,6,-4,g,Y6,R,The,Phi);
  cube T06m3 = getT(0,6,-3,g,Y6,R,The,Phi);  cube T16m3 = getT(1,6,-3,g,Y6,R,The,Phi);  cube T26m3 = getT(2,6,-3,g,Y6,R,The,Phi);  cube T36m3 = getT(3,6,-3,g,Y6,R,The,Phi);
  cube T06m2 = getT(0,6,-2,g,Y6,R,The,Phi);  cube T16m2 = getT(1,6,-2,g,Y6,R,The,Phi);  cube T26m2 = getT(2,6,-2,g,Y6,R,The,Phi);  cube T36m2 = getT(3,6,-2,g,Y6,R,The,Phi);
  cube T06m1 = getT(0,6,-1,g,Y6,R,The,Phi);  cube T16m1 = getT(1,6,-1,g,Y6,R,The,Phi);  cube T26m1 = getT(2,6,-1,g,Y6,R,The,Phi);  cube T36m1 = getT(3,6,-1,g,Y6,R,The,Phi);
  cube T060  = getT(0,6,+0,g,Y6,R,The,Phi);  cube T160  = getT(1,6,+0,g,Y6,R,The,Phi);  cube T260  = getT(2,6,+0,g,Y6,R,The,Phi);  cube T360  = getT(3,6,+0,g,Y6,R,The,Phi);
  cube T06p1 = getT(0,6,+1,g,Y6,R,The,Phi);  cube T16p1 = getT(1,6,+1,g,Y6,R,The,Phi);  cube T26p1 = getT(2,6,+1,g,Y6,R,The,Phi);  cube T36p1 = getT(3,6,+1,g,Y6,R,The,Phi);
  cube T06p2 = getT(0,6,+2,g,Y6,R,The,Phi);  cube T16p2 = getT(1,6,+2,g,Y6,R,The,Phi);  cube T26p2 = getT(2,6,+2,g,Y6,R,The,Phi);  cube T36p2 = getT(3,6,+2,g,Y6,R,The,Phi);
  cube T06p3 = getT(0,6,+3,g,Y6,R,The,Phi);  cube T16p3 = getT(1,6,+3,g,Y6,R,The,Phi);  cube T26p3 = getT(2,6,+3,g,Y6,R,The,Phi);  cube T36p3 = getT(3,6,+3,g,Y6,R,The,Phi);
  cube T06p4 = getT(0,6,+4,g,Y6,R,The,Phi);  cube T16p4 = getT(1,6,+4,g,Y6,R,The,Phi);  cube T26p4 = getT(2,6,+4,g,Y6,R,The,Phi);  cube T36p4 = getT(3,6,+4,g,Y6,R,The,Phi);
  cube T06p5 = getT(0,6,+5,g,Y6,R,The,Phi);  cube T16p5 = getT(1,6,+5,g,Y6,R,The,Phi);  cube T26p5 = getT(2,6,+5,g,Y6,R,The,Phi);  cube T36p5 = getT(3,6,+5,g,Y6,R,The,Phi);
  cube T06p6 = getT(0,6,+6,g,Y6,R,The,Phi);  cube T16p6 = getT(1,6,+6,g,Y6,R,The,Phi);  cube T26p6 = getT(2,6,+6,g,Y6,R,The,Phi);  cube T36p6 = getT(3,6,+6,g,Y6,R,The,Phi);
  cube T07m7 = getT(0,7,-7,g,Y7,R,The,Phi);  cube T17m7 = getT(1,7,-7,g,Y7,R,The,Phi);  cube T27m7 = getT(2,7,-7,g,Y7,R,The,Phi);  cube T37m7 = getT(3,7,-7,g,Y7,R,The,Phi);
  cube T07m6 = getT(0,7,-6,g,Y7,R,The,Phi);  cube T17m6 = getT(1,7,-6,g,Y7,R,The,Phi);  cube T27m6 = getT(2,7,-6,g,Y7,R,The,Phi);  cube T37m6 = getT(3,7,-6,g,Y7,R,The,Phi);
  cube T07m5 = getT(0,7,-5,g,Y7,R,The,Phi);  cube T17m5 = getT(1,7,-5,g,Y7,R,The,Phi);  cube T27m5 = getT(2,7,-5,g,Y7,R,The,Phi);  cube T37m5 = getT(3,7,-5,g,Y7,R,The,Phi);
  cube T07m4 = getT(0,7,-4,g,Y7,R,The,Phi);  cube T17m4 = getT(1,7,-4,g,Y7,R,The,Phi);  cube T27m4 = getT(2,7,-4,g,Y7,R,The,Phi);  cube T37m4 = getT(3,7,-4,g,Y7,R,The,Phi);
  cube T07m3 = getT(0,7,-3,g,Y7,R,The,Phi);  cube T17m3 = getT(1,7,-3,g,Y7,R,The,Phi);  cube T27m3 = getT(2,7,-3,g,Y7,R,The,Phi);  cube T37m3 = getT(3,7,-3,g,Y7,R,The,Phi);
  cube T07m2 = getT(0,7,-2,g,Y7,R,The,Phi);  cube T17m2 = getT(1,7,-2,g,Y7,R,The,Phi);  cube T27m2 = getT(2,7,-2,g,Y7,R,The,Phi);  cube T37m2 = getT(3,7,-2,g,Y7,R,The,Phi);
  cube T07m1 = getT(0,7,-1,g,Y7,R,The,Phi);  cube T17m1 = getT(1,7,-1,g,Y7,R,The,Phi);  cube T27m1 = getT(2,7,-1,g,Y7,R,The,Phi);  cube T37m1 = getT(3,7,-1,g,Y7,R,The,Phi);
  cube T070  = getT(0,7,+0,g,Y7,R,The,Phi);  cube T170  = getT(1,7,+0,g,Y7,R,The,Phi);  cube T270  = getT(2,7,+0,g,Y7,R,The,Phi);  cube T370  = getT(3,7,+0,g,Y7,R,The,Phi);
  cube T07p1 = getT(0,7,+1,g,Y7,R,The,Phi);  cube T17p1 = getT(1,7,+1,g,Y7,R,The,Phi);  cube T27p1 = getT(2,7,+1,g,Y7,R,The,Phi);  cube T37p1 = getT(3,7,+1,g,Y7,R,The,Phi);
  cube T07p2 = getT(0,7,+2,g,Y7,R,The,Phi);  cube T17p2 = getT(1,7,+2,g,Y7,R,The,Phi);  cube T27p2 = getT(2,7,+2,g,Y7,R,The,Phi);  cube T37p2 = getT(3,7,+2,g,Y7,R,The,Phi);
  cube T07p3 = getT(0,7,+3,g,Y7,R,The,Phi);  cube T17p3 = getT(1,7,+3,g,Y7,R,The,Phi);  cube T27p3 = getT(2,7,+3,g,Y7,R,The,Phi);  cube T37p3 = getT(3,7,+3,g,Y7,R,The,Phi);
  cube T07p4 = getT(0,7,+4,g,Y7,R,The,Phi);  cube T17p4 = getT(1,7,+4,g,Y7,R,The,Phi);  cube T27p4 = getT(2,7,+4,g,Y7,R,The,Phi);  cube T37p4 = getT(3,7,+4,g,Y7,R,The,Phi);
  cube T07p5 = getT(0,7,+5,g,Y7,R,The,Phi);  cube T17p5 = getT(1,7,+5,g,Y7,R,The,Phi);  cube T27p5 = getT(2,7,+5,g,Y7,R,The,Phi);  cube T37p5 = getT(3,7,+5,g,Y7,R,The,Phi);
  cube T07p6 = getT(0,7,+6,g,Y7,R,The,Phi);  cube T17p6 = getT(1,7,+6,g,Y7,R,The,Phi);  cube T27p6 = getT(2,7,+6,g,Y7,R,The,Phi);  cube T37p6 = getT(3,7,+6,g,Y7,R,The,Phi);
  cube T07p7 = getT(0,7,+7,g,Y7,R,The,Phi);  cube T17p7 = getT(1,7,+7,g,Y7,R,The,Phi);  cube T27p7 = getT(2,7,+7,g,Y7,R,The,Phi);  cube T37p7 = getT(3,7,+7,g,Y7,R,The,Phi);
  cube T08m8 = getT(0,8,-8,g,Y8,R,The,Phi);  cube T18m8 = getT(1,8,-8,g,Y8,R,The,Phi);  cube T28m8 = getT(2,8,-8,g,Y8,R,The,Phi);  cube T38m8 = getT(3,8,-8,g,Y8,R,The,Phi);
  cube T08m7 = getT(0,8,-7,g,Y8,R,The,Phi);  cube T18m7 = getT(1,8,-7,g,Y8,R,The,Phi);  cube T28m7 = getT(2,8,-7,g,Y8,R,The,Phi);  cube T38m7 = getT(3,8,-7,g,Y8,R,The,Phi);
  cube T08m6 = getT(0,8,-6,g,Y8,R,The,Phi);  cube T18m6 = getT(1,8,-6,g,Y8,R,The,Phi);  cube T28m6 = getT(2,8,-6,g,Y8,R,The,Phi);  cube T38m6 = getT(3,8,-6,g,Y8,R,The,Phi);
  cube T08m5 = getT(0,8,-5,g,Y8,R,The,Phi);  cube T18m5 = getT(1,8,-5,g,Y8,R,The,Phi);  cube T28m5 = getT(2,8,-5,g,Y8,R,The,Phi);  cube T38m5 = getT(3,8,-5,g,Y8,R,The,Phi);
  cube T08m4 = getT(0,8,-4,g,Y8,R,The,Phi);  cube T18m4 = getT(1,8,-4,g,Y8,R,The,Phi);  cube T28m4 = getT(2,8,-4,g,Y8,R,The,Phi);  cube T38m4 = getT(3,8,-4,g,Y8,R,The,Phi);
  cube T08m3 = getT(0,8,-3,g,Y8,R,The,Phi);  cube T18m3 = getT(1,8,-3,g,Y8,R,The,Phi);  cube T28m3 = getT(2,8,-3,g,Y8,R,The,Phi);  cube T38m3 = getT(3,8,-3,g,Y8,R,The,Phi);
  cube T08m2 = getT(0,8,-2,g,Y8,R,The,Phi);  cube T18m2 = getT(1,8,-2,g,Y8,R,The,Phi);  cube T28m2 = getT(2,8,-2,g,Y8,R,The,Phi);  cube T38m2 = getT(3,8,-2,g,Y8,R,The,Phi);
  cube T08m1 = getT(0,8,-1,g,Y8,R,The,Phi);  cube T18m1 = getT(1,8,-1,g,Y8,R,The,Phi);  cube T28m1 = getT(2,8,-1,g,Y8,R,The,Phi);  cube T38m1 = getT(3,8,-1,g,Y8,R,The,Phi);
  cube T080  = getT(0,8,+0,g,Y8,R,The,Phi);  cube T180  = getT(1,8,+0,g,Y8,R,The,Phi);  cube T280  = getT(2,8,+0,g,Y8,R,The,Phi);  cube T380  = getT(3,8,+0,g,Y8,R,The,Phi);
  cube T08p1 = getT(0,8,+1,g,Y8,R,The,Phi);  cube T18p1 = getT(1,8,+1,g,Y8,R,The,Phi);  cube T28p1 = getT(2,8,+1,g,Y8,R,The,Phi);  cube T38p1 = getT(3,8,+1,g,Y8,R,The,Phi);
  cube T08p2 = getT(0,8,+2,g,Y8,R,The,Phi);  cube T18p2 = getT(1,8,+2,g,Y8,R,The,Phi);  cube T28p2 = getT(2,8,+2,g,Y8,R,The,Phi);  cube T38p2 = getT(3,8,+2,g,Y8,R,The,Phi);
  cube T08p3 = getT(0,8,+3,g,Y8,R,The,Phi);  cube T18p3 = getT(1,8,+3,g,Y8,R,The,Phi);  cube T28p3 = getT(2,8,+3,g,Y8,R,The,Phi);  cube T38p3 = getT(3,8,+3,g,Y8,R,The,Phi);
  cube T08p4 = getT(0,8,+4,g,Y8,R,The,Phi);  cube T18p4 = getT(1,8,+4,g,Y8,R,The,Phi);  cube T28p4 = getT(2,8,+4,g,Y8,R,The,Phi);  cube T38p4 = getT(3,8,+4,g,Y8,R,The,Phi);
  cube T08p5 = getT(0,8,+5,g,Y8,R,The,Phi);  cube T18p5 = getT(1,8,+5,g,Y8,R,The,Phi);  cube T28p5 = getT(2,8,+5,g,Y8,R,The,Phi);  cube T38p5 = getT(3,8,+5,g,Y8,R,The,Phi);
  cube T08p6 = getT(0,8,+6,g,Y8,R,The,Phi);  cube T18p6 = getT(1,8,+6,g,Y8,R,The,Phi);  cube T28p6 = getT(2,8,+6,g,Y8,R,The,Phi);  cube T38p6 = getT(3,8,+6,g,Y8,R,The,Phi);
  cube T08p7 = getT(0,8,+7,g,Y8,R,The,Phi);  cube T18p7 = getT(1,8,+7,g,Y8,R,The,Phi);  cube T28p7 = getT(2,8,+7,g,Y8,R,The,Phi);  cube T38p7 = getT(3,8,+7,g,Y8,R,The,Phi);
  cube T08p8 = getT(0,8,+8,g,Y8,R,The,Phi);  cube T18p8 = getT(1,8,+8,g,Y8,R,The,Phi);  cube T28p8 = getT(2,8,+8,g,Y8,R,The,Phi);  cube T38p8 = getT(3,8,+8,g,Y8,R,The,Phi);
  cube T09m9 = getT(0,9,-9,g,Y9,R,The,Phi);  cube T19m9 = getT(1,9,-9,g,Y9,R,The,Phi);  cube T29m9 = getT(2,9,-9,g,Y9,R,The,Phi);  cube T39m9 = getT(3,9,-9,g,Y9,R,The,Phi);
  cube T09m8 = getT(0,9,-8,g,Y9,R,The,Phi);  cube T19m8 = getT(1,9,-8,g,Y9,R,The,Phi);  cube T29m8 = getT(2,9,-8,g,Y9,R,The,Phi);  cube T39m8 = getT(3,9,-8,g,Y9,R,The,Phi);
  cube T09m7 = getT(0,9,-7,g,Y9,R,The,Phi);  cube T19m7 = getT(1,9,-7,g,Y9,R,The,Phi);  cube T29m7 = getT(2,9,-7,g,Y9,R,The,Phi);  cube T39m7 = getT(3,9,-7,g,Y9,R,The,Phi);
  cube T09m6 = getT(0,9,-6,g,Y9,R,The,Phi);  cube T19m6 = getT(1,9,-6,g,Y9,R,The,Phi);  cube T29m6 = getT(2,9,-6,g,Y9,R,The,Phi);  cube T39m6 = getT(3,9,-6,g,Y9,R,The,Phi);
  cube T09m5 = getT(0,9,-5,g,Y9,R,The,Phi);  cube T19m5 = getT(1,9,-5,g,Y9,R,The,Phi);  cube T29m5 = getT(2,9,-5,g,Y9,R,The,Phi);  cube T39m5 = getT(3,9,-5,g,Y9,R,The,Phi);
  cube T09m4 = getT(0,9,-4,g,Y9,R,The,Phi);  cube T19m4 = getT(1,9,-4,g,Y9,R,The,Phi);  cube T29m4 = getT(2,9,-4,g,Y9,R,The,Phi);  cube T39m4 = getT(3,9,-4,g,Y9,R,The,Phi);
  cube T09m3 = getT(0,9,-3,g,Y9,R,The,Phi);  cube T19m3 = getT(1,9,-3,g,Y9,R,The,Phi);  cube T29m3 = getT(2,9,-3,g,Y9,R,The,Phi);  cube T39m3 = getT(3,9,-3,g,Y9,R,The,Phi);
  cube T09m2 = getT(0,9,-2,g,Y9,R,The,Phi);  cube T19m2 = getT(1,9,-2,g,Y9,R,The,Phi);  cube T29m2 = getT(2,9,-2,g,Y9,R,The,Phi);  cube T39m2 = getT(3,9,-2,g,Y9,R,The,Phi);
  cube T09m1 = getT(0,9,-1,g,Y9,R,The,Phi);  cube T19m1 = getT(1,9,-1,g,Y9,R,The,Phi);  cube T29m1 = getT(2,9,-1,g,Y9,R,The,Phi);  cube T39m1 = getT(3,9,-1,g,Y9,R,The,Phi);
  cube T090  = getT(0,9,+0,g,Y9,R,The,Phi);  cube T190  = getT(1,9,+0,g,Y9,R,The,Phi);  cube T290  = getT(2,9,+0,g,Y9,R,The,Phi);  cube T390  = getT(3,9,+0,g,Y9,R,The,Phi);
  cube T09p1 = getT(0,9,+1,g,Y9,R,The,Phi);  cube T19p1 = getT(1,9,+1,g,Y9,R,The,Phi);  cube T29p1 = getT(2,9,+1,g,Y9,R,The,Phi);  cube T39p1 = getT(3,9,+1,g,Y9,R,The,Phi);
  cube T09p2 = getT(0,9,+2,g,Y9,R,The,Phi);  cube T19p2 = getT(1,9,+2,g,Y9,R,The,Phi);  cube T29p2 = getT(2,9,+2,g,Y9,R,The,Phi);  cube T39p2 = getT(3,9,+2,g,Y9,R,The,Phi);
  cube T09p3 = getT(0,9,+3,g,Y9,R,The,Phi);  cube T19p3 = getT(1,9,+3,g,Y9,R,The,Phi);  cube T29p3 = getT(2,9,+3,g,Y9,R,The,Phi);  cube T39p3 = getT(3,9,+3,g,Y9,R,The,Phi);
  cube T09p4 = getT(0,9,+4,g,Y9,R,The,Phi);  cube T19p4 = getT(1,9,+4,g,Y9,R,The,Phi);  cube T29p4 = getT(2,9,+4,g,Y9,R,The,Phi);  cube T39p4 = getT(3,9,+4,g,Y9,R,The,Phi);
  cube T09p5 = getT(0,9,+5,g,Y9,R,The,Phi);  cube T19p5 = getT(1,9,+5,g,Y9,R,The,Phi);  cube T29p5 = getT(2,9,+5,g,Y9,R,The,Phi);  cube T39p5 = getT(3,9,+5,g,Y9,R,The,Phi);
  cube T09p6 = getT(0,9,+6,g,Y9,R,The,Phi);  cube T19p6 = getT(1,9,+6,g,Y9,R,The,Phi);  cube T29p6 = getT(2,9,+6,g,Y9,R,The,Phi);  cube T39p6 = getT(3,9,+6,g,Y9,R,The,Phi);
  cube T09p7 = getT(0,9,+7,g,Y9,R,The,Phi);  cube T19p7 = getT(1,9,+7,g,Y9,R,The,Phi);  cube T29p7 = getT(2,9,+7,g,Y9,R,The,Phi);  cube T39p7 = getT(3,9,+7,g,Y9,R,The,Phi);
  cube T09p8 = getT(0,9,+8,g,Y9,R,The,Phi);  cube T19p8 = getT(1,9,+8,g,Y9,R,The,Phi);  cube T29p8 = getT(2,9,+8,g,Y9,R,The,Phi);  cube T39p8 = getT(3,9,+8,g,Y9,R,The,Phi);
  cube T09p9 = getT(0,9,+9,g,Y9,R,The,Phi);  cube T19p9 = getT(1,9,+9,g,Y9,R,The,Phi);  cube T29p9 = getT(2,9,+9,g,Y9,R,The,Phi);  cube T39p9 = getT(3,9,+9,g,Y9,R,The,Phi);

  cout << "Part 3: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 4) get coefs c_nlm by Integration -> Gauss quaduature used in 3D. Int Tnml(r,The,Phi) rho(r,theta,phi) dV
  //----------------------------------------------------------------------------------------------------------------

  cube intMe = rho%GLC;
//  GLC.save("GLC.bi");

  double C000=rsc*integ3D(intMe,T000);  double C100=rsc*integ3D(intMe,T100);  double C200=rsc*integ3D(intMe,T200);  double C300=rsc*integ3D(intMe,T300);
  double C01m1=rsc*integ3D(intMe,T01m1);double C11m1=rsc*integ3D(intMe,T11m1);double C21m1=rsc*integ3D(intMe,T21m1);  double C31m1=rsc*integ3D(intMe,T31m1);
  double C010 =rsc*integ3D(intMe,T010 ); double C110 =rsc*integ3D(intMe,T110 );double C210 =rsc*integ3D(intMe,T210 );  double C310 =rsc*integ3D(intMe,T310 );
  double C01p1=rsc*integ3D(intMe,T01p1);double C11p1=rsc*integ3D(intMe,T11p1);double C21p1=rsc*integ3D(intMe,T21p1);  double C31p1=rsc*integ3D(intMe,T31p1);
  double C02m2=rsc*integ3D(intMe,T02m2);double C12m2=rsc*integ3D(intMe,T12m2);double C22m2=rsc*integ3D(intMe,T22m2);  double C32m2=rsc*integ3D(intMe,T32m2);
  double C02m1=rsc*integ3D(intMe,T02m1);double C12m1=rsc*integ3D(intMe,T12m1);double C22m1=rsc*integ3D(intMe,T22m1);  double C32m1=rsc*integ3D(intMe,T32m1);
  double C020 =rsc*integ3D(intMe,T020 ); double C120 =rsc*integ3D(intMe,T120 );double C220 =rsc*integ3D(intMe,T220 );  double C320 =rsc*integ3D(intMe,T320 );
  double C02p1=rsc*integ3D(intMe,T02p1);double C12p1=rsc*integ3D(intMe,T12p1);double C22p1=rsc*integ3D(intMe,T22p1);  double C32p1=rsc*integ3D(intMe,T32p1);
  double C02p2=rsc*integ3D(intMe,T02p2);double C12p2=rsc*integ3D(intMe,T12p2);double C22p2=rsc*integ3D(intMe,T22p2);  double C32p2=rsc*integ3D(intMe,T32p2);
  double C03m3=rsc*integ3D(intMe,T03m3);double C13m3=rsc*integ3D(intMe,T13m3);double C23m3=rsc*integ3D(intMe,T23m3);  double C33m3=rsc*integ3D(intMe,T33m3);
  double C03m2=rsc*integ3D(intMe,T03m2);double C13m2=rsc*integ3D(intMe,T13m2);double C23m2=rsc*integ3D(intMe,T23m2);  double C33m2=rsc*integ3D(intMe,T33m2);
  double C03m1=rsc*integ3D(intMe,T03m1);double C13m1=rsc*integ3D(intMe,T13m1);double C23m1=rsc*integ3D(intMe,T23m1);  double C33m1=rsc*integ3D(intMe,T33m1);
  double C030 =rsc*integ3D(intMe,T030 ); double C130 =rsc*integ3D(intMe,T130 );double C230 =rsc*integ3D(intMe,T230 );  double C330 =rsc*integ3D(intMe,T330 );
  double C03p1=rsc*integ3D(intMe,T03p1);double C13p1=rsc*integ3D(intMe,T13p1);double C23p1=rsc*integ3D(intMe,T23p1);  double C33p1=rsc*integ3D(intMe,T33p1);
  double C03p2=rsc*integ3D(intMe,T03p2);double C13p2=rsc*integ3D(intMe,T13p2);double C23p2=rsc*integ3D(intMe,T23p2);  double C33p2=rsc*integ3D(intMe,T33p2);
  double C03p3=rsc*integ3D(intMe,T03p3);double C13p3=rsc*integ3D(intMe,T13p3);double C23p3=rsc*integ3D(intMe,T23p3);  double C33p3=rsc*integ3D(intMe,T33p3);
  double C04m4=rsc*integ3D(intMe,T04m4);double C14m4=rsc*integ3D(intMe,T14m4);double C24m4=rsc*integ3D(intMe,T24m4);  double C34m4=rsc*integ3D(intMe,T34m4);
  double C04m3=rsc*integ3D(intMe,T04m3);double C14m3=rsc*integ3D(intMe,T14m3);double C24m3=rsc*integ3D(intMe,T24m3);  double C34m3=rsc*integ3D(intMe,T34m3);
  double C04m2=rsc*integ3D(intMe,T04m2);double C14m2=rsc*integ3D(intMe,T14m2);double C24m2=rsc*integ3D(intMe,T24m2);  double C34m2=rsc*integ3D(intMe,T34m2);
  double C04m1=rsc*integ3D(intMe,T04m1);double C14m1=rsc*integ3D(intMe,T14m1);double C24m1=rsc*integ3D(intMe,T24m1);  double C34m1=rsc*integ3D(intMe,T34m1);
  double C040 =rsc*integ3D(intMe,T040 ); double C140 =rsc*integ3D(intMe,T140 );double C240 =rsc*integ3D(intMe,T240 );  double C340 =rsc*integ3D(intMe,T340 );
  double C04p1=rsc*integ3D(intMe,T04p1);double C14p1=rsc*integ3D(intMe,T14p1);double C24p1=rsc*integ3D(intMe,T24p1);  double C34p1=rsc*integ3D(intMe,T34p1);
  double C04p2=rsc*integ3D(intMe,T04p2);double C14p2=rsc*integ3D(intMe,T14p2);double C24p2=rsc*integ3D(intMe,T24p2);  double C34p2=rsc*integ3D(intMe,T34p2);
  double C04p3=rsc*integ3D(intMe,T04p3);double C14p3=rsc*integ3D(intMe,T14p3);double C24p3=rsc*integ3D(intMe,T24p3);  double C34p3=rsc*integ3D(intMe,T34p3);
  double C04p4=rsc*integ3D(intMe,T04p4);double C14p4=rsc*integ3D(intMe,T14p4);double C24p4=rsc*integ3D(intMe,T24p4);  double C34p4=rsc*integ3D(intMe,T34p4);
  double C05m5=rsc*integ3D(intMe,T05m5);double C15m5=rsc*integ3D(intMe,T15m5);double C25m5=rsc*integ3D(intMe,T25m5);  double C35m5=rsc*integ3D(intMe,T35m5);
  double C05m4=rsc*integ3D(intMe,T05m4);double C15m4=rsc*integ3D(intMe,T15m4);double C25m4=rsc*integ3D(intMe,T25m4);  double C35m4=rsc*integ3D(intMe,T35m4);
  double C05m3=rsc*integ3D(intMe,T05m3);double C15m3=rsc*integ3D(intMe,T15m3);double C25m3=rsc*integ3D(intMe,T25m3);  double C35m3=rsc*integ3D(intMe,T35m3);
  double C05m2=rsc*integ3D(intMe,T05m2);double C15m2=rsc*integ3D(intMe,T15m2);double C25m2=rsc*integ3D(intMe,T25m2);  double C35m2=rsc*integ3D(intMe,T35m2);
  double C05m1=rsc*integ3D(intMe,T05m1);double C15m1=rsc*integ3D(intMe,T15m1);double C25m1=rsc*integ3D(intMe,T25m1);  double C35m1=rsc*integ3D(intMe,T35m1);
  double C050 =rsc*integ3D(intMe,T050 ); double C150 =rsc*integ3D(intMe,T150 );double C250 =rsc*integ3D(intMe,T250 );  double C350 =rsc*integ3D(intMe,T350 );
  double C05p1=rsc*integ3D(intMe,T05p1);double C15p1=rsc*integ3D(intMe,T15p1);double C25p1=rsc*integ3D(intMe,T25p1);  double C35p1=rsc*integ3D(intMe,T35p1);
  double C05p2=rsc*integ3D(intMe,T05p2);double C15p2=rsc*integ3D(intMe,T15p2);double C25p2=rsc*integ3D(intMe,T25p2);  double C35p2=rsc*integ3D(intMe,T35p2);
  double C05p3=rsc*integ3D(intMe,T05p3);double C15p3=rsc*integ3D(intMe,T15p3);double C25p3=rsc*integ3D(intMe,T25p3);  double C35p3=rsc*integ3D(intMe,T35p3);
  double C05p4=rsc*integ3D(intMe,T05p4);double C15p4=rsc*integ3D(intMe,T15p4);double C25p4=rsc*integ3D(intMe,T25p4);  double C35p4=rsc*integ3D(intMe,T35p4);
  double C05p5=rsc*integ3D(intMe,T05p5);double C15p5=rsc*integ3D(intMe,T15p5);double C25p5=rsc*integ3D(intMe,T25p5);  double C35p5=rsc*integ3D(intMe,T35p5);
  double C06m6=rsc*integ3D(intMe,T06m6);double C16m6=rsc*integ3D(intMe,T16m6);double C26m6=rsc*integ3D(intMe,T26m6);  double C36m6=rsc*integ3D(intMe,T36m6);
  double C06m5=rsc*integ3D(intMe,T06m5);double C16m5=rsc*integ3D(intMe,T16m5);double C26m5=rsc*integ3D(intMe,T26m5);  double C36m5=rsc*integ3D(intMe,T36m5);
  double C06m4=rsc*integ3D(intMe,T06m4);double C16m4=rsc*integ3D(intMe,T16m4);double C26m4=rsc*integ3D(intMe,T26m4);  double C36m4=rsc*integ3D(intMe,T36m4);
  double C06m3=rsc*integ3D(intMe,T06m3);double C16m3=rsc*integ3D(intMe,T16m3);double C26m3=rsc*integ3D(intMe,T26m3);  double C36m3=rsc*integ3D(intMe,T36m3);
  double C06m2=rsc*integ3D(intMe,T06m2);double C16m2=rsc*integ3D(intMe,T16m2);double C26m2=rsc*integ3D(intMe,T26m2);  double C36m2=rsc*integ3D(intMe,T36m2);
  double C06m1=rsc*integ3D(intMe,T06m1);double C16m1=rsc*integ3D(intMe,T16m1);double C26m1=rsc*integ3D(intMe,T26m1);  double C36m1=rsc*integ3D(intMe,T36m1);
  double C060 =rsc*integ3D(intMe,T060 ); double C160 =rsc*integ3D(intMe,T160 );double C260 =rsc*integ3D(intMe,T260 );  double C360 =rsc*integ3D(intMe,T360 );
  double C06p1=rsc*integ3D(intMe,T06p1);double C16p1=rsc*integ3D(intMe,T16p1);double C26p1=rsc*integ3D(intMe,T26p1);  double C36p1=rsc*integ3D(intMe,T36p1);
  double C06p2=rsc*integ3D(intMe,T06p2);double C16p2=rsc*integ3D(intMe,T16p2);double C26p2=rsc*integ3D(intMe,T26p2);  double C36p2=rsc*integ3D(intMe,T36p2);
  double C06p3=rsc*integ3D(intMe,T06p3);double C16p3=rsc*integ3D(intMe,T16p3);double C26p3=rsc*integ3D(intMe,T26p3);  double C36p3=rsc*integ3D(intMe,T36p3);
  double C06p4=rsc*integ3D(intMe,T06p4);double C16p4=rsc*integ3D(intMe,T16p4);double C26p4=rsc*integ3D(intMe,T26p4);  double C36p4=rsc*integ3D(intMe,T36p4);
  double C06p5=rsc*integ3D(intMe,T06p5);double C16p5=rsc*integ3D(intMe,T16p5);double C26p5=rsc*integ3D(intMe,T26p5);  double C36p5=rsc*integ3D(intMe,T36p5);
  double C06p6=rsc*integ3D(intMe,T06p6);double C16p6=rsc*integ3D(intMe,T16p6);double C26p6=rsc*integ3D(intMe,T26p6);  double C36p6=rsc*integ3D(intMe,T36p6);
  double C07m7=rsc*integ3D(intMe,T07m7);double C17m7=rsc*integ3D(intMe,T17m7);double C27m7=rsc*integ3D(intMe,T27m7);  double C37m7=rsc*integ3D(intMe,T37m7);
  double C07m6=rsc*integ3D(intMe,T07m6);double C17m6=rsc*integ3D(intMe,T17m6);double C27m6=rsc*integ3D(intMe,T27m6);  double C37m6=rsc*integ3D(intMe,T37m6);
  double C07m5=rsc*integ3D(intMe,T07m5);double C17m5=rsc*integ3D(intMe,T17m5);double C27m5=rsc*integ3D(intMe,T27m5);  double C37m5=rsc*integ3D(intMe,T37m5);
  double C07m4=rsc*integ3D(intMe,T07m4);double C17m4=rsc*integ3D(intMe,T17m4);double C27m4=rsc*integ3D(intMe,T27m4);  double C37m4=rsc*integ3D(intMe,T37m4);
  double C07m3=rsc*integ3D(intMe,T07m3);double C17m3=rsc*integ3D(intMe,T17m3);double C27m3=rsc*integ3D(intMe,T27m3);  double C37m3=rsc*integ3D(intMe,T37m3);
  double C07m2=rsc*integ3D(intMe,T07m2);double C17m2=rsc*integ3D(intMe,T17m2);double C27m2=rsc*integ3D(intMe,T27m2);  double C37m2=rsc*integ3D(intMe,T37m2);
  double C07m1=rsc*integ3D(intMe,T07m1);double C17m1=rsc*integ3D(intMe,T17m1);double C27m1=rsc*integ3D(intMe,T27m1);  double C37m1=rsc*integ3D(intMe,T37m1);
  double C070 =rsc*integ3D(intMe,T070 ); double C170 =rsc*integ3D(intMe,T170 );double C270 =rsc*integ3D(intMe,T270 );  double C370 =rsc*integ3D(intMe,T370 );
  double C07p1=rsc*integ3D(intMe,T07p1);double C17p1=rsc*integ3D(intMe,T17p1);double C27p1=rsc*integ3D(intMe,T27p1);  double C37p1=rsc*integ3D(intMe,T37p1);
  double C07p2=rsc*integ3D(intMe,T07p2);double C17p2=rsc*integ3D(intMe,T17p2);double C27p2=rsc*integ3D(intMe,T27p2);  double C37p2=rsc*integ3D(intMe,T37p2);
  double C07p3=rsc*integ3D(intMe,T07p3);double C17p3=rsc*integ3D(intMe,T17p3);double C27p3=rsc*integ3D(intMe,T27p3);  double C37p3=rsc*integ3D(intMe,T37p3);
  double C07p4=rsc*integ3D(intMe,T07p4);double C17p4=rsc*integ3D(intMe,T17p4);double C27p4=rsc*integ3D(intMe,T27p4);  double C37p4=rsc*integ3D(intMe,T37p4);
  double C07p5=rsc*integ3D(intMe,T07p5);double C17p5=rsc*integ3D(intMe,T17p5);double C27p5=rsc*integ3D(intMe,T27p5);  double C37p5=rsc*integ3D(intMe,T37p5);
  double C07p6=rsc*integ3D(intMe,T07p6);double C17p6=rsc*integ3D(intMe,T17p6);double C27p6=rsc*integ3D(intMe,T27p6);  double C37p6=rsc*integ3D(intMe,T37p6);
  double C07p7=rsc*integ3D(intMe,T07p7);double C17p7=rsc*integ3D(intMe,T17p7);double C27p7=rsc*integ3D(intMe,T27p7);  double C37p7=rsc*integ3D(intMe,T37p7);
  double C08m8=rsc*integ3D(intMe,T08m8); double C18m8=rsc*integ3D(intMe,T18m8); double C28m8=rsc*integ3D(intMe,T28m8);  double C38m8=rsc*integ3D(intMe,T38m8);
  double C08m7=rsc*integ3D(intMe,T08m7); double C18m7=rsc*integ3D(intMe,T18m7); double C28m7=rsc*integ3D(intMe,T28m7);  double C38m7=rsc*integ3D(intMe,T38m7);
  double C08m6=rsc*integ3D(intMe,T08m6); double C18m6=rsc*integ3D(intMe,T18m6); double C28m6=rsc*integ3D(intMe,T28m6);  double C38m6=rsc*integ3D(intMe,T38m6);
  double C08m5=rsc*integ3D(intMe,T08m5); double C18m5=rsc*integ3D(intMe,T18m5); double C28m5=rsc*integ3D(intMe,T28m5);  double C38m5=rsc*integ3D(intMe,T38m5);
  double C08m4=rsc*integ3D(intMe,T08m4); double C18m4=rsc*integ3D(intMe,T18m4); double C28m4=rsc*integ3D(intMe,T28m4);  double C38m4=rsc*integ3D(intMe,T38m4);
  double C08m3=rsc*integ3D(intMe,T08m3); double C18m3=rsc*integ3D(intMe,T18m3); double C28m3=rsc*integ3D(intMe,T28m3);  double C38m3=rsc*integ3D(intMe,T38m3);
  double C08m2=rsc*integ3D(intMe,T08m2); double C18m2=rsc*integ3D(intMe,T18m2); double C28m2=rsc*integ3D(intMe,T28m2);  double C38m2=rsc*integ3D(intMe,T38m2);
  double C08m1=rsc*integ3D(intMe,T08m1); double C18m1=rsc*integ3D(intMe,T18m1); double C28m1=rsc*integ3D(intMe,T28m1);  double C38m1=rsc*integ3D(intMe,T38m1);
  double C080 =rsc*integ3D(intMe,T080 );  double C180 =rsc*integ3D(intMe,T180 ); double C280 =rsc*integ3D(intMe,T280 );  double C380 =rsc*integ3D(intMe,T380 );
  double C08p1=rsc*integ3D(intMe,T08p1); double C18p1=rsc*integ3D(intMe,T18p1); double C28p1=rsc*integ3D(intMe,T28p1);  double C38p1=rsc*integ3D(intMe,T38p1);
  double C08p2=rsc*integ3D(intMe,T08p2); double C18p2=rsc*integ3D(intMe,T18p2); double C28p2=rsc*integ3D(intMe,T28p2);  double C38p2=rsc*integ3D(intMe,T38p2);
  double C08p3=rsc*integ3D(intMe,T08p3); double C18p3=rsc*integ3D(intMe,T18p3); double C28p3=rsc*integ3D(intMe,T28p3);  double C38p3=rsc*integ3D(intMe,T38p3);
  double C08p4=rsc*integ3D(intMe,T08p4); double C18p4=rsc*integ3D(intMe,T18p4); double C28p4=rsc*integ3D(intMe,T28p4);  double C38p4=rsc*integ3D(intMe,T38p4);
  double C08p5=rsc*integ3D(intMe,T08p5); double C18p5=rsc*integ3D(intMe,T18p5); double C28p5=rsc*integ3D(intMe,T28p5);  double C38p5=rsc*integ3D(intMe,T38p5);
  double C08p6=rsc*integ3D(intMe,T08p6); double C18p6=rsc*integ3D(intMe,T18p6); double C28p6=rsc*integ3D(intMe,T28p6);  double C38p6=rsc*integ3D(intMe,T38p6);
  double C08p7=rsc*integ3D(intMe,T08p7); double C18p7=rsc*integ3D(intMe,T18p7); double C28p7=rsc*integ3D(intMe,T28p7);  double C38p7=rsc*integ3D(intMe,T38p7);
  double C08p8=rsc*integ3D(intMe,T08p8); double C18p8=rsc*integ3D(intMe,T18p8); double C28p8=rsc*integ3D(intMe,T28p8);  double C38p8=rsc*integ3D(intMe,T38p8);
  double C09m9=rsc*integ3D(intMe,T09m9); double C19m9=rsc*integ3D(intMe,T19m9); double C29m9=rsc*integ3D(intMe,T29m9);  double C39m9=rsc*integ3D(intMe,T39m9);
  double C09m8=rsc*integ3D(intMe,T09m8); double C19m8=rsc*integ3D(intMe,T19m8); double C29m8=rsc*integ3D(intMe,T29m8);  double C39m8=rsc*integ3D(intMe,T39m8);
  double C09m7=rsc*integ3D(intMe,T09m7); double C19m7=rsc*integ3D(intMe,T19m7); double C29m7=rsc*integ3D(intMe,T29m7);  double C39m7=rsc*integ3D(intMe,T39m7);
  double C09m6=rsc*integ3D(intMe,T09m6); double C19m6=rsc*integ3D(intMe,T19m6); double C29m6=rsc*integ3D(intMe,T29m6);  double C39m6=rsc*integ3D(intMe,T39m6);
  double C09m5=rsc*integ3D(intMe,T09m5); double C19m5=rsc*integ3D(intMe,T19m5); double C29m5=rsc*integ3D(intMe,T29m5);  double C39m5=rsc*integ3D(intMe,T39m5);
  double C09m4=rsc*integ3D(intMe,T09m4); double C19m4=rsc*integ3D(intMe,T19m4); double C29m4=rsc*integ3D(intMe,T29m4);  double C39m4=rsc*integ3D(intMe,T39m4);
  double C09m3=rsc*integ3D(intMe,T09m3); double C19m3=rsc*integ3D(intMe,T19m3); double C29m3=rsc*integ3D(intMe,T29m3);  double C39m3=rsc*integ3D(intMe,T39m3);
  double C09m2=rsc*integ3D(intMe,T09m2); double C19m2=rsc*integ3D(intMe,T19m2); double C29m2=rsc*integ3D(intMe,T29m2);  double C39m2=rsc*integ3D(intMe,T39m2);
  double C09m1=rsc*integ3D(intMe,T09m1); double C19m1=rsc*integ3D(intMe,T19m1); double C29m1=rsc*integ3D(intMe,T29m1);  double C39m1=rsc*integ3D(intMe,T39m1);
  double C090 =rsc*integ3D(intMe,T090 );  double C190 =rsc*integ3D(intMe,T190 ); double C290 =rsc*integ3D(intMe,T290 );  double C390 =rsc*integ3D(intMe,T390 );
  double C09p1=rsc*integ3D(intMe,T09p1); double C19p1=rsc*integ3D(intMe,T19p1); double C29p1=rsc*integ3D(intMe,T29p1);  double C39p1=rsc*integ3D(intMe,T39p1);
  double C09p2=rsc*integ3D(intMe,T09p2); double C19p2=rsc*integ3D(intMe,T19p2); double C29p2=rsc*integ3D(intMe,T29p2);  double C39p2=rsc*integ3D(intMe,T39p2);
  double C09p3=rsc*integ3D(intMe,T09p3); double C19p3=rsc*integ3D(intMe,T19p3); double C29p3=rsc*integ3D(intMe,T29p3);  double C39p3=rsc*integ3D(intMe,T39p3);
  double C09p4=rsc*integ3D(intMe,T09p4); double C19p4=rsc*integ3D(intMe,T19p4); double C29p4=rsc*integ3D(intMe,T29p4);  double C39p4=rsc*integ3D(intMe,T39p4);
  double C09p5=rsc*integ3D(intMe,T09p5); double C19p5=rsc*integ3D(intMe,T19p5); double C29p5=rsc*integ3D(intMe,T29p5);  double C39p5=rsc*integ3D(intMe,T39p5);
  double C09p6=rsc*integ3D(intMe,T09p6); double C19p6=rsc*integ3D(intMe,T19p6); double C29p6=rsc*integ3D(intMe,T29p6);  double C39p6=rsc*integ3D(intMe,T39p6);
  double C09p7=rsc*integ3D(intMe,T09p7); double C19p7=rsc*integ3D(intMe,T19p7); double C29p7=rsc*integ3D(intMe,T29p7);  double C39p7=rsc*integ3D(intMe,T39p7);
  double C09p8=rsc*integ3D(intMe,T09p8); double C19p8=rsc*integ3D(intMe,T19p8); double C29p8=rsc*integ3D(intMe,T29p8);  double C39p8=rsc*integ3D(intMe,T39p8);
  double C09p9=rsc*integ3D(intMe,T09p9); double C19p9=rsc*integ3D(intMe,T19p9); double C29p9=rsc*integ3D(intMe,T29p9);  double C39p9=rsc*integ3D(intMe,T39p9);

//double C[3][10][21];
//memset(C, 0.0, sizeof C);
  mat C = zeros<mat>(4,100);
    
 // Saving the values in C[n][l]. 
  C(0,0) = C000 ; C(1,0) = C100; C(2,0) =  C200;  C(3,0) = C300 ;
  C(0,1) = C01m1 ; C(1,1) = C11m1; C(2,1) =  C21m1;  C(3,1) = C31m1 ;
  C(0,2) = C010  ; C(1,2) = C110 ; C(2,2) =  C210 ;  C(3,2) = C310  ;
  C(0,3) = C01p1 ; C(1,3) = C11p1; C(2,3) =  C21p1;  C(3,3) = C31p1 ;
  C(0,4) = C02m2 ; C(1,4) = C12m2; C(2,4) =  C22m2;  C(3,4) = C32m2 ;
  C(0,5) = C02m1 ; C(1,5) = C12m1; C(2,5) =  C22m1;  C(3,5) = C32m1 ;
  C(0,6) = C020  ; C(1,6) = C120 ; C(2,6) =  C220 ;  C(3,6) = C320  ;
  C(0,7) = C02p1 ; C(1,7) = C12p1; C(2,7) =  C22p1;  C(3,7) = C32p1 ;
  C(0,8) = C02p2 ; C(1,8) = C12p2; C(2,8) =  C22p2;  C(3,8) = C32p2 ;
  C(0,9) = C03m3 ; C(1,9) = C13m3; C(2,9) =  C23m3;  C(3,9) = C33m3 ;
  C(0,10) = C03m2 ; C(1,10) = C13m2; C(2,10) =  C23m2;  C(3,10) = C33m2 ;
  C(0,11) = C03m1 ; C(1,11) = C13m1; C(2,11) =  C23m1;  C(3,11) = C33m1 ;
  C(0,12) = C030  ; C(1,12) = C130 ; C(2,12) =  C230 ;  C(3,12) = C330  ;
  C(0,13) = C03p1 ; C(1,13) = C13p1; C(2,13) =  C23p1;  C(3,13) = C33p1 ;
  C(0,14) = C03p2 ; C(1,14) = C13p2; C(2,14) =  C23p2;  C(3,14) = C33p2 ;
  C(0,15) = C03p3 ; C(1,15) = C13p3; C(2,15) =  C23p3;  C(3,15) = C33p3 ;
  C(0,16) = C04m4 ; C(1,16) = C14m4; C(2,16) =  C24m4;  C(3,16) = C34m4 ;
  C(0,17) = C04m3 ; C(1,17) = C14m3; C(2,17) =  C24m3;  C(3,17) = C34m3 ;
  C(0,18) = C04m2 ; C(1,18) = C14m2; C(2,18) =  C24m2;  C(3,18) = C34m2 ;
  C(0,19) = C04m1 ; C(1,19) = C14m1; C(2,19) =  C24m1;  C(3,19) = C34m1 ;
  C(0,20) = C040  ; C(1,20) = C140 ; C(2,20) =  C240 ;  C(3,20) = C340  ;
  C(0,21) = C04p1 ; C(1,21) = C14p1; C(2,21) =  C24p1;  C(3,21) = C34p1 ;
  C(0,22) = C04p2 ; C(1,22) = C14p2; C(2,22) =  C24p2;  C(3,22) = C34p2 ;
  C(0,23) = C04p3 ; C(1,23) = C14p3; C(2,23) =  C24p3;  C(3,23) = C34p3 ;
  C(0,24) = C04p4 ; C(1,24) = C14p4; C(2,24) =  C24p4;  C(3,24) = C34p4 ;
  C(0,25) = C05m5 ; C(1,25) = C15m5; C(2,25) =  C25m5;  C(3,25) = C35m5 ;
  C(0,26) = C05m4 ; C(1,26) = C15m4; C(2,26) =  C25m4;  C(3,26) = C35m4 ;
  C(0,27) = C05m3 ; C(1,27) = C15m3; C(2,27) =  C25m3;  C(3,27) = C35m3 ;
  C(0,28) = C05m2 ; C(1,28) = C15m2; C(2,28) =  C25m2;  C(3,28) = C35m2 ;
  C(0,29) = C05m1 ; C(1,29) = C15m1; C(2,29) =  C25m1;  C(3,29) = C35m1 ;
  C(0,30) = C050  ; C(1,30) = C150 ; C(2,30) =  C250 ;  C(3,30) = C350  ;
  C(0,31) = C05p1 ; C(1,31) = C15p1; C(2,31) =  C25p1;  C(3,31) = C35p1 ;
  C(0,32) = C05p2 ; C(1,32) = C15p2; C(2,32) =  C25p2;  C(3,32) = C35p2 ;
  C(0,33) = C05p3 ; C(1,33) = C15p3; C(2,33) =  C25p3;  C(3,33) = C35p3 ;
  C(0,34) = C05p4 ; C(1,34) = C15p4; C(2,34) =  C25p4;  C(3,34) = C35p4 ;
  C(0,35) = C05p5 ; C(1,35) = C15p5; C(2,35) =  C25p5;  C(3,35) = C35p5 ;
  C(0,36) = C06m6 ; C(1,36) = C16m6; C(2,36) =  C26m6;  C(3,36) = C36m6 ;
  C(0,37) = C06m5 ; C(1,37) = C16m5; C(2,37) =  C26m5;  C(3,37) = C36m5 ;
  C(0,38) = C06m4 ; C(1,38) = C16m4; C(2,38) =  C26m4;  C(3,38) = C36m4 ;
  C(0,39) = C06m3 ; C(1,39) = C16m3; C(2,39) =  C26m3;  C(3,39) = C36m3 ;
  C(0,40) = C06m2 ; C(1,40) = C16m2; C(2,40) =  C26m2;  C(3,40) = C36m2 ;
  C(0,41) = C06m1 ; C(1,41) = C16m1; C(2,41) =  C26m1;  C(3,41) = C36m1 ;
  C(0,42) = C060  ; C(1,42) = C160 ; C(2,42) =  C260 ;  C(3,42) = C360  ;
  C(0,43) = C06p1 ; C(1,43) = C16p1; C(2,43) =  C26p1;  C(3,43) = C36p1 ;
  C(0,44) = C06p2 ; C(1,44) = C16p2; C(2,44) =  C26p2;  C(3,44) = C36p2 ;
  C(0,45) = C06p3 ; C(1,45) = C16p3; C(2,45) =  C26p3;  C(3,45) = C36p3 ;
  C(0,46) = C06p4 ; C(1,46) = C16p4; C(2,46) =  C26p4;  C(3,46) = C36p4 ;
  C(0,47) = C06p5 ; C(1,47) = C16p5; C(2,47) =  C26p5;  C(3,47) = C36p5 ;
  C(0,48) = C06p6 ; C(1,48) = C16p6; C(2,48) =  C26p6;  C(3,48) = C36p6 ;
  C(0,49) = C07m7 ; C(1,49) = C17m7; C(2,49) =  C27m7;  C(3,49) = C37m7 ;
  C(0,50) = C07m6 ; C(1,50) = C17m6; C(2,50) =  C27m6;  C(3,50) = C37m6 ;
  C(0,51) = C07m5 ; C(1,51) = C17m5; C(2,51) =  C27m5;  C(3,51) = C37m5 ;
  C(0,52) = C07m4 ; C(1,52) = C17m4; C(2,52) =  C27m4;  C(3,52) = C37m4 ;
  C(0,53) = C07m3 ; C(1,53) = C17m3; C(2,53) =  C27m3;  C(3,53) = C37m3 ;
  C(0,54) = C07m2 ; C(1,54) = C17m2; C(2,54) =  C27m2;  C(3,54) = C37m2 ;
  C(0,55) = C07m1 ; C(1,55) = C17m1; C(2,55) =  C27m1;  C(3,55) = C37m1 ;
  C(0,56) = C070  ; C(1,56) = C170 ; C(2,56) =  C270 ;  C(3,56) = C370  ;
  C(0,57) = C07p1 ; C(1,57) = C17p1; C(2,57) =  C27p1;  C(3,57) = C37p1 ;
  C(0,58) = C07p2 ; C(1,58) = C17p2; C(2,58) =  C27p2;  C(3,58) = C37p2 ;
  C(0,59) = C07p3 ; C(1,59) = C17p3; C(2,59) =  C27p3;  C(3,59) = C37p3 ;
  C(0,60) = C07p4 ; C(1,60) = C17p4; C(2,60) =  C27p4;  C(3,60) = C37p4 ;
  C(0,61) = C07p5 ; C(1,61) = C17p5; C(2,61) =  C27p5;  C(3,61) = C37p5 ;
  C(0,62) = C07p6 ; C(1,62) = C17p6; C(2,62) =  C27p6;  C(3,62) = C37p6 ;
  C(0,63) = C07p7 ; C(1,63) = C17p7; C(2,63) =  C27p7;  C(3,63) = C37p7 ;
  C(0,64) = C08m8 ; C(1,64) = C18m8; C(2,64) =  C28m8;  C(3,64) = C38m8 ;
  C(0,65) = C08m7 ; C(1,65) = C18m7; C(2,65) =  C28m7;  C(3,65) = C38m7 ;
  C(0,66) = C08m6 ; C(1,66) = C18m6; C(2,66) =  C28m6;  C(3,66) = C38m6 ;
  C(0,67) = C08m5 ; C(1,67) = C18m5; C(2,67) =  C28m5;  C(3,67) = C38m5 ;
  C(0,68) = C08m4 ; C(1,68) = C18m4; C(2,68) =  C28m4;  C(3,68) = C38m4 ;
  C(0,69) = C08m3 ; C(1,69) = C18m3; C(2,69) =  C28m3;  C(3,69) = C38m3 ;
  C(0,70) = C08m2 ; C(1,70) = C18m2; C(2,70) =  C28m2;  C(3,70) = C38m2 ;
  C(0,71) = C08m1 ; C(1,71) = C18m1; C(2,71) =  C28m1;  C(3,71) = C38m1 ;
  C(0,72) = C080  ; C(1,72) = C180 ; C(2,72) =  C280 ;  C(3,72) = C380  ;
  C(0,73) = C08p1 ; C(1,73) = C18p1; C(2,73) =  C28p1;  C(3,73) = C38p1 ;
  C(0,74) = C08p2 ; C(1,74) = C18p2; C(2,74) =  C28p2;  C(3,74) = C38p2 ;
  C(0,75) = C08p3 ; C(1,75) = C18p3; C(2,75) =  C28p3;  C(3,75) = C38p3 ;
  C(0,76) = C08p4 ; C(1,76) = C18p4; C(2,76) =  C28p4;  C(3,76) = C38p4 ;
  C(0,77) = C08p5 ; C(1,77) = C18p5; C(2,77) =  C28p5;  C(3,77) = C38p5 ;
  C(0,78) = C08p6 ; C(1,78) = C18p6; C(2,78) =  C28p6;  C(3,78) = C38p6 ;
  C(0,79) = C08p7 ; C(1,79) = C18p7; C(2,79) =  C28p7;  C(3,79) = C38p7 ;
  C(0,80) = C08p8 ; C(1,80) = C18p8; C(2,80) =  C28p8;  C(3,80) = C38p8 ;
  C(0,81) = C09m9 ; C(1,81) = C19m9; C(2,81) =  C29m9;  C(3,81) = C39m9 ;
  C(0,82) = C09m8 ; C(1,82) = C19m8; C(2,82) =  C29m8;  C(3,82) = C39m8 ;
  C(0,83) = C09m7 ; C(1,83) = C19m7; C(2,83) =  C29m7;  C(3,83) = C39m7 ;
  C(0,84) = C09m6 ; C(1,84) = C19m6; C(2,84) =  C29m6;  C(3,84) = C39m6 ;
  C(0,85) = C09m5 ; C(1,85) = C19m5; C(2,85) =  C29m5;  C(3,85) = C39m5 ;
  C(0,86) = C09m4 ; C(1,86) = C19m4; C(2,86) =  C29m4;  C(3,86) = C39m4 ;
  C(0,87) = C09m3 ; C(1,87) = C19m3; C(2,87) =  C29m3;  C(3,87) = C39m3 ;
  C(0,88) = C09m2 ; C(1,88) = C19m2; C(2,88) =  C29m2;  C(3,88) = C39m2 ;
  C(0,89) = C09m1 ; C(1,89) = C19m1; C(2,89) =  C29m1;  C(3,89) = C39m1 ;
  C(0,90) = C090  ; C(1,90) = C190 ; C(2,90) =  C290 ;  C(3,90) = C390  ;
  C(0,91) = C09p1 ; C(1,91) = C19p1; C(2,91) =  C29p1;  C(3,91) = C39p1 ;
  C(0,92) = C09p2 ; C(1,92) = C19p2; C(2,92) =  C29p2;  C(3,92) = C39p2 ;
  C(0,93) = C09p3 ; C(1,93) = C19p3; C(2,93) =  C29p3;  C(3,93) = C39p3 ;
  C(0,94) = C09p4 ; C(1,94) = C19p4; C(2,94) =  C29p4;  C(3,94) = C39p4 ;
  C(0,95) = C09p5 ; C(1,95) = C19p5; C(2,95) =  C29p5;  C(3,95) = C39p5 ;
  C(0,96) = C09p6 ; C(1,96) = C19p6; C(2,96) =  C29p6;  C(3,96) = C39p6 ;
  C(0,97) = C09p7 ; C(1,97) = C19p7; C(2,97) =  C29p7;  C(3,97) = C39p7 ;
  C(0,98) = C09p8 ; C(1,98) = C19p8; C(2,98) =  C29p8;  C(3,98) = C39p8 ;
  C(0,99) = C09p9 ; C(1,99) = C19p9; C(2,99) =  C29p9;  C(3,99) = C39p9 ;

//C.t().print(" ");

  cout << "Part 2: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 5) get Power Spectrum-> Gauss quaduature used in 3D. Int Tnml(r,The,Phi) rho(r,theta,phi) dV
  //----------------------------------------------------------------------------------------------------------------
  

 int checkN = 0; 
 double ppp = 0;
 cube P = zeros<cube>(4,4,10);
  for(int n1=0; n1 < 4; n1++){ 
    for(int n2=0; n2 < 4; n2++){ 
        checkN = 0;
      for(int l=0; l <= 9; l++){ 
      for(int m=-l; m <= l; m++){ 
        P(n1,n2,l) += C(n1,checkN)*C(n2,checkN);
        checkN++;
        }

      }
    }
  }

  cout << size(C) << endl;
  cout << "Part 5: Done" << endl;

  
P.print("P");


return 0;
}



































