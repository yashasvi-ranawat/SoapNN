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

  double rcut = 10.0;
  double rsc = pi*pi*0.5*0.5*rcut; // rescaleing the integration for gauss-legendre quaduature.

  double sig = 1;

  double ao = 1.0;
  double z = 1.0;
  double norm = pow(sqrt(z/ao),3);
  
  //----------------------------------------------------------------------------------------------------------------
  // Part 1) Retrieving Data -> W, Gauss-Legendre, XYZ-Smeared
  //----------------------------------------------------------------------------------------------------------------
  
//  mat W; // [APB eq.25]
//  W.load("WMatData/W3.csv");

  mat GL; // [http://keisan.casio.com/exec/system/1329114617 (June 5th 2017)] , produced by Octave. W(:,0) -> GL coord. pos. W(:,1) -> GL weights.
//  GL.load("parameters100.txt");
//  GL.load("parameters70.txt");
//  GL.load("parameters50.txt");
  GL.load("P200_both.txt");

  vec R = rcut*0.5*GL.col(0) + rcut*0.5 ; // rescaled R for gauss-legendre quaduature
  vec The = pi*GL.col(0)*0.5 + pi*0.5;  // rescaled The for gauss-legendre quaduature
  vec Phi = pi*GL.col(0) + pi;            // rescaled Phi for gauss-legendre quaduature

  cube X;
  cube Y;
  cube Z;
//  X.load("X.bi");
//  Y.load("Y.bi");
//  Z.load("Z.bi");


  X = getSphericalToCartCubeX( R, The, Phi);
  Y = getSphericalToCartCubeY( R, The, Phi);
  Z = getSphericalToCartCubeZ( R, The, Phi);
  X.save("X200.bi");
  Y.save("Y200.bi");
  Z.save("Z200.bi");

  mat coord = getPos(argv[1]);
  string* type = getType(argv[1]);

  vec typeA = zeros<vec>(coord.n_rows);
  vec typeB = zeros<vec>(coord.n_rows);

  coord = posAve(coord); 
//  coord.print("Before:");
//  coord = rotate3d(coord,1,1,1);
//  coord.print("After:");
//  coord(coord.n_rows - 1,2) += 0.1;

  for(int i=0; i < coord.n_rows; i++)  { 
     if(type[i] == "Mo" ){typeA(i) =1;}
   }
  for(int i=0; i < coord.n_rows; i++)  { 
     if(type[i] == "S" ){typeB(i) =1;}
   }
  mat coord_a = zeros<mat>(sum(typeA) + 1,3);
  mat coord_b = zeros<mat>(sum(typeB) + 1,3);
  double newJ = 0;
  for(int i=0; i < coord.n_rows; i++)  { 
     if(type[i] == "Mo" ){coord_a.row(newJ) = coord.row(i); newJ++;}
   }
  newJ = 0;
  for(int i=0; i < coord.n_rows; i++)  { 
     if(type[i] == "S" ){coord_b.row(newJ) = coord.row(i); newJ++;}
   }


  coord_a.row(coord_a.n_rows - 1) = coord.row(coord.n_rows - 1);
  coord_b.row(coord_b.n_rows - 1) = coord.row(coord.n_rows - 1);


  cube rho_a = getGaussDistr(coord_a,R, The, Phi, X, Y, Z, sig);
cout << size(Y) << endl;
  cube rho_b = getGaussDistr(coord_b,R, The, Phi, X, Y, Z, sig);


//  coord_a.print("A");
//  coord_b.print("B");

//  cout << "FFF" << endl;
//  rho.print("rho");

//  mat printRho(100,100);
//  for(int i=0; i < 100; i++) 
//    for(int j=0; j < 100; j++)    printRho(i,j) = rho.at(15,i,j); // sanity check for Gaussian distribution of atoms.

  vec lastAtom = coord.row(coord.n_rows - 1).t();

  cube GLC(GL.n_rows,GL.n_rows,GL.n_rows);
//  GLC.load("GLC.bi");
  for(int i=0; i < GL.n_rows; i++){ 
    for(int j=0; j < GL.n_rows; j++){ 
      for(int k=0; k < GL.n_rows; k++){ 

        GLC.at(i,j,k) = GL.at(i,1)*GL.at(j,1)*GL.at(k,1); // Setting up the GL weights.

      }
    }
  }

  GLC.save("GLC200.bi");

//  cout << "Part 1: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 2) Constructing Basis Functions -> g_n(r), N_a, phi_a(r),  
  //----------------------------------------------------------------------------------------------------------------
  
  mat gn(R.n_rows,4);// Radial Basis Functions [APB eq.25]. g(*,:) -> n's. g(:,*) -> r's of GL coord. pos. Slater used.
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
   
//  cout << "Part 2: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 3) Preparing Integrand -> Cube Tnlm(r,The,Phi) Due to memory problem, the T's and C's are split into many
  // variables;
  //----------------------------------------------------------------------------------------------------------------

  cube T000 = getTMat(0,0,0,g,Y0,R,The,Phi); cube T100 = getTMat(1,0,0,g,Y0,R,The,Phi); cube T200 = getTMat(2,0,0,g,Y0,R,The,Phi); cube T300 = getTMat(3,0,0,g,Y0,R,The,Phi);
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

//  cout << "Part 3: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 4) get coefs c_nlm by Integration -> Gauss quaduature used in 3D. Int Tnml(r,The,Phi) rho(r,theta,phi) dV
  //----------------------------------------------------------------------------------------------------------------

  cube intMea = rho_a%GLC;
  cube intMeb = rho_b%GLC;

  double Ca000=rsc*integ3D(intMea,T000);  double Ca100=rsc*integ3D(intMea,T100);  double Ca200=rsc*integ3D(intMea,T200);    double Ca300=rsc*integ3D(intMea,T300);
  double Cb000=rsc*integ3D(intMeb,T000);  double Cb100=rsc*integ3D(intMeb,T100);  double Cb200=rsc*integ3D(intMeb,T200);    double Cb300=rsc*integ3D(intMeb,T300);
  double Ca01m1=rsc*integ3D(intMea,T01m1);double Ca11m1=rsc*integ3D(intMea,T11m1);double Ca21m1=rsc*integ3D(intMea,T21m1);  double Ca31m1=rsc*integ3D(intMea,T31m1);
  double Cb01m1=rsc*integ3D(intMeb,T01m1);double Cb11m1=rsc*integ3D(intMeb,T11m1);double Cb21m1=rsc*integ3D(intMeb,T21m1);  double Cb31m1=rsc*integ3D(intMeb,T31m1);
  double Ca010 =rsc*integ3D(intMea,T010 ); double Ca110 =rsc*integ3D(intMea,T110 );double Ca210 =rsc*integ3D(intMea,T210 );  double Ca310 =rsc*integ3D(intMea,T310 );
  double Cb010 =rsc*integ3D(intMeb,T010 ); double Cb110 =rsc*integ3D(intMeb,T110 );double Cb210 =rsc*integ3D(intMeb,T210 );  double Cb310 =rsc*integ3D(intMeb,T310 );
  double Ca01p1=rsc*integ3D(intMea,T01p1);double Ca11p1=rsc*integ3D(intMea,T11p1);double Ca21p1=rsc*integ3D(intMea,T21p1);  double Ca31p1=rsc*integ3D(intMea,T31p1);
  double Cb01p1=rsc*integ3D(intMeb,T01p1);double Cb11p1=rsc*integ3D(intMeb,T11p1);double Cb21p1=rsc*integ3D(intMeb,T21p1);  double Cb31p1=rsc*integ3D(intMeb,T31p1);
  double Ca02m2=rsc*integ3D(intMea,T02m2);double Ca12m2=rsc*integ3D(intMea,T12m2);double Ca22m2=rsc*integ3D(intMea,T22m2);  double Ca32m2=rsc*integ3D(intMea,T32m2);
  double Cb02m2=rsc*integ3D(intMeb,T02m2);double Cb12m2=rsc*integ3D(intMeb,T12m2);double Cb22m2=rsc*integ3D(intMeb,T22m2);  double Cb32m2=rsc*integ3D(intMeb,T32m2);
  double Ca02m1=rsc*integ3D(intMea,T02m1);double Ca12m1=rsc*integ3D(intMea,T12m1);double Ca22m1=rsc*integ3D(intMea,T22m1);  double Ca32m1=rsc*integ3D(intMea,T32m1);
  double Cb02m1=rsc*integ3D(intMeb,T02m1);double Cb12m1=rsc*integ3D(intMeb,T12m1);double Cb22m1=rsc*integ3D(intMeb,T22m1);  double Cb32m1=rsc*integ3D(intMeb,T32m1);
  double Ca020 =rsc*integ3D(intMea,T020 ); double Ca120 =rsc*integ3D(intMea,T120 );double Ca220 =rsc*integ3D(intMea,T220 );  double Ca320 =rsc*integ3D(intMea,T320 );
  double Cb020 =rsc*integ3D(intMeb,T020 ); double Cb120 =rsc*integ3D(intMeb,T120 );double Cb220 =rsc*integ3D(intMeb,T220 );  double Cb320 =rsc*integ3D(intMeb,T320 );
  double Ca02p1=rsc*integ3D(intMea,T02p1);double Ca12p1=rsc*integ3D(intMea,T12p1);double Ca22p1=rsc*integ3D(intMea,T22p1);  double Ca32p1=rsc*integ3D(intMea,T32p1);
  double Cb02p1=rsc*integ3D(intMeb,T02p1);double Cb12p1=rsc*integ3D(intMeb,T12p1);double Cb22p1=rsc*integ3D(intMeb,T22p1);  double Cb32p1=rsc*integ3D(intMeb,T32p1);
  double Ca02p2=rsc*integ3D(intMea,T02p2);double Ca12p2=rsc*integ3D(intMea,T12p2);double Ca22p2=rsc*integ3D(intMea,T22p2);  double Ca32p2=rsc*integ3D(intMea,T32p2);
  double Cb02p2=rsc*integ3D(intMeb,T02p2);double Cb12p2=rsc*integ3D(intMeb,T12p2);double Cb22p2=rsc*integ3D(intMeb,T22p2);  double Cb32p2=rsc*integ3D(intMeb,T32p2);
  double Ca03m3=rsc*integ3D(intMea,T03m3);double Ca13m3=rsc*integ3D(intMea,T13m3);double Ca23m3=rsc*integ3D(intMea,T23m3);  double Ca33m3=rsc*integ3D(intMea,T33m3);
  double Cb03m3=rsc*integ3D(intMeb,T03m3);double Cb13m3=rsc*integ3D(intMeb,T13m3);double Cb23m3=rsc*integ3D(intMeb,T23m3);  double Cb33m3=rsc*integ3D(intMeb,T33m3);
  double Ca03m2=rsc*integ3D(intMea,T03m2);double Ca13m2=rsc*integ3D(intMea,T13m2);double Ca23m2=rsc*integ3D(intMea,T23m2);  double Ca33m2=rsc*integ3D(intMea,T33m2);
  double Cb03m2=rsc*integ3D(intMeb,T03m2);double Cb13m2=rsc*integ3D(intMeb,T13m2);double Cb23m2=rsc*integ3D(intMeb,T23m2);  double Cb33m2=rsc*integ3D(intMeb,T33m2);
  double Ca03m1=rsc*integ3D(intMea,T03m1);double Ca13m1=rsc*integ3D(intMea,T13m1);double Ca23m1=rsc*integ3D(intMea,T23m1);  double Ca33m1=rsc*integ3D(intMea,T33m1);
  double Cb03m1=rsc*integ3D(intMeb,T03m1);double Cb13m1=rsc*integ3D(intMeb,T13m1);double Cb23m1=rsc*integ3D(intMeb,T23m1);  double Cb33m1=rsc*integ3D(intMeb,T33m1);
  double Ca030 =rsc*integ3D(intMea,T030 ); double Ca130 =rsc*integ3D(intMea,T130 );double Ca230 =rsc*integ3D(intMea,T230 );  double Ca330 =rsc*integ3D(intMea,T330 );
  double Cb030 =rsc*integ3D(intMeb,T030 ); double Cb130 =rsc*integ3D(intMeb,T130 );double Cb230 =rsc*integ3D(intMeb,T230 );  double Cb330 =rsc*integ3D(intMeb,T330 );
  double Ca03p1=rsc*integ3D(intMea,T03p1);double Ca13p1=rsc*integ3D(intMea,T13p1);double Ca23p1=rsc*integ3D(intMea,T23p1);  double Ca33p1=rsc*integ3D(intMea,T33p1);
  double Cb03p1=rsc*integ3D(intMeb,T03p1);double Cb13p1=rsc*integ3D(intMeb,T13p1);double Cb23p1=rsc*integ3D(intMeb,T23p1);  double Cb33p1=rsc*integ3D(intMeb,T33p1);
  double Ca03p2=rsc*integ3D(intMea,T03p2);double Ca13p2=rsc*integ3D(intMea,T13p2);double Ca23p2=rsc*integ3D(intMea,T23p2);  double Ca33p2=rsc*integ3D(intMea,T33p2);
  double Cb03p2=rsc*integ3D(intMeb,T03p2);double Cb13p2=rsc*integ3D(intMeb,T13p2);double Cb23p2=rsc*integ3D(intMeb,T23p2);  double Cb33p2=rsc*integ3D(intMeb,T33p2);
  double Ca03p3=rsc*integ3D(intMea,T03p3);double Ca13p3=rsc*integ3D(intMea,T13p3);double Ca23p3=rsc*integ3D(intMea,T23p3);  double Ca33p3=rsc*integ3D(intMea,T33p3);
  double Cb03p3=rsc*integ3D(intMeb,T03p3);double Cb13p3=rsc*integ3D(intMeb,T13p3);double Cb23p3=rsc*integ3D(intMeb,T23p3);  double Cb33p3=rsc*integ3D(intMeb,T33p3);
  double Ca04m4=rsc*integ3D(intMea,T04m4);double Ca14m4=rsc*integ3D(intMea,T14m4);double Ca24m4=rsc*integ3D(intMea,T24m4);  double Ca34m4=rsc*integ3D(intMea,T34m4);
  double Cb04m4=rsc*integ3D(intMeb,T04m4);double Cb14m4=rsc*integ3D(intMeb,T14m4);double Cb24m4=rsc*integ3D(intMeb,T24m4);  double Cb34m4=rsc*integ3D(intMeb,T34m4);
  double Ca04m3=rsc*integ3D(intMea,T04m3);double Ca14m3=rsc*integ3D(intMea,T14m3);double Ca24m3=rsc*integ3D(intMea,T24m3);  double Ca34m3=rsc*integ3D(intMea,T34m3);
  double Cb04m3=rsc*integ3D(intMeb,T04m3);double Cb14m3=rsc*integ3D(intMeb,T14m3);double Cb24m3=rsc*integ3D(intMeb,T24m3);  double Cb34m3=rsc*integ3D(intMeb,T34m3);
  double Ca04m2=rsc*integ3D(intMea,T04m2);double Ca14m2=rsc*integ3D(intMea,T14m2);double Ca24m2=rsc*integ3D(intMea,T24m2);  double Ca34m2=rsc*integ3D(intMea,T34m2);
  double Cb04m2=rsc*integ3D(intMeb,T04m2);double Cb14m2=rsc*integ3D(intMeb,T14m2);double Cb24m2=rsc*integ3D(intMeb,T24m2);  double Cb34m2=rsc*integ3D(intMeb,T34m2);
  double Ca04m1=rsc*integ3D(intMea,T04m1);double Ca14m1=rsc*integ3D(intMea,T14m1);double Ca24m1=rsc*integ3D(intMea,T24m1);  double Ca34m1=rsc*integ3D(intMea,T34m1);
  double Cb04m1=rsc*integ3D(intMeb,T04m1);double Cb14m1=rsc*integ3D(intMeb,T14m1);double Cb24m1=rsc*integ3D(intMeb,T24m1);  double Cb34m1=rsc*integ3D(intMeb,T34m1);
  double Ca040 =rsc*integ3D(intMea,T040 ); double Ca140 =rsc*integ3D(intMea,T140 );double Ca240 =rsc*integ3D(intMea,T240 );  double Ca340 =rsc*integ3D(intMea,T340 );
  double Cb040 =rsc*integ3D(intMeb,T040 ); double Cb140 =rsc*integ3D(intMeb,T140 );double Cb240 =rsc*integ3D(intMeb,T240 );  double Cb340 =rsc*integ3D(intMeb,T340 );
  double Ca04p1=rsc*integ3D(intMea,T04p1);double Ca14p1=rsc*integ3D(intMea,T14p1);double Ca24p1=rsc*integ3D(intMea,T24p1);  double Ca34p1=rsc*integ3D(intMea,T34p1);
  double Cb04p1=rsc*integ3D(intMeb,T04p1);double Cb14p1=rsc*integ3D(intMeb,T14p1);double Cb24p1=rsc*integ3D(intMeb,T24p1);  double Cb34p1=rsc*integ3D(intMeb,T34p1);
  double Ca04p2=rsc*integ3D(intMea,T04p2);double Ca14p2=rsc*integ3D(intMea,T14p2);double Ca24p2=rsc*integ3D(intMea,T24p2);  double Ca34p2=rsc*integ3D(intMea,T34p2);
  double Cb04p2=rsc*integ3D(intMeb,T04p2);double Cb14p2=rsc*integ3D(intMeb,T14p2);double Cb24p2=rsc*integ3D(intMeb,T24p2);  double Cb34p2=rsc*integ3D(intMeb,T34p2);
  double Ca04p3=rsc*integ3D(intMea,T04p3);double Ca14p3=rsc*integ3D(intMea,T14p3);double Ca24p3=rsc*integ3D(intMea,T24p3);  double Ca34p3=rsc*integ3D(intMea,T34p3);
  double Cb04p3=rsc*integ3D(intMeb,T04p3);double Cb14p3=rsc*integ3D(intMeb,T14p3);double Cb24p3=rsc*integ3D(intMeb,T24p3);  double Cb34p3=rsc*integ3D(intMeb,T34p3);
  double Ca04p4=rsc*integ3D(intMea,T04p4);double Ca14p4=rsc*integ3D(intMea,T14p4);double Ca24p4=rsc*integ3D(intMea,T24p4);  double Ca34p4=rsc*integ3D(intMea,T34p4);
  double Cb04p4=rsc*integ3D(intMeb,T04p4);double Cb14p4=rsc*integ3D(intMeb,T14p4);double Cb24p4=rsc*integ3D(intMeb,T24p4);  double Cb34p4=rsc*integ3D(intMeb,T34p4);
  double Ca05m5=rsc*integ3D(intMea,T05m5);double Ca15m5=rsc*integ3D(intMea,T15m5);double Ca25m5=rsc*integ3D(intMea,T25m5);  double Ca35m5=rsc*integ3D(intMea,T35m5);
  double Cb05m5=rsc*integ3D(intMeb,T05m5);double Cb15m5=rsc*integ3D(intMeb,T15m5);double Cb25m5=rsc*integ3D(intMeb,T25m5);  double Cb35m5=rsc*integ3D(intMeb,T35m5);
  double Ca05m4=rsc*integ3D(intMea,T05m4);double Ca15m4=rsc*integ3D(intMea,T15m4);double Ca25m4=rsc*integ3D(intMea,T25m4);  double Ca35m4=rsc*integ3D(intMea,T35m4);
  double Cb05m4=rsc*integ3D(intMeb,T05m4);double Cb15m4=rsc*integ3D(intMeb,T15m4);double Cb25m4=rsc*integ3D(intMeb,T25m4);  double Cb35m4=rsc*integ3D(intMeb,T35m4);
  double Ca05m3=rsc*integ3D(intMea,T05m3);double Ca15m3=rsc*integ3D(intMea,T15m3);double Ca25m3=rsc*integ3D(intMea,T25m3);  double Ca35m3=rsc*integ3D(intMea,T35m3);
  double Cb05m3=rsc*integ3D(intMeb,T05m3);double Cb15m3=rsc*integ3D(intMeb,T15m3);double Cb25m3=rsc*integ3D(intMeb,T25m3);  double Cb35m3=rsc*integ3D(intMeb,T35m3);
  double Ca05m2=rsc*integ3D(intMea,T05m2);double Ca15m2=rsc*integ3D(intMea,T15m2);double Ca25m2=rsc*integ3D(intMea,T25m2);  double Ca35m2=rsc*integ3D(intMea,T35m2);
  double Cb05m2=rsc*integ3D(intMeb,T05m2);double Cb15m2=rsc*integ3D(intMeb,T15m2);double Cb25m2=rsc*integ3D(intMeb,T25m2);  double Cb35m2=rsc*integ3D(intMeb,T35m2);
  double Ca05m1=rsc*integ3D(intMea,T05m1);double Ca15m1=rsc*integ3D(intMea,T15m1);double Ca25m1=rsc*integ3D(intMea,T25m1);  double Ca35m1=rsc*integ3D(intMea,T35m1);
  double Cb05m1=rsc*integ3D(intMeb,T05m1);double Cb15m1=rsc*integ3D(intMeb,T15m1);double Cb25m1=rsc*integ3D(intMeb,T25m1);  double Cb35m1=rsc*integ3D(intMeb,T35m1);
  double Ca050 =rsc*integ3D(intMea,T050 ); double Ca150 =rsc*integ3D(intMea,T150 );double Ca250 =rsc*integ3D(intMea,T250 );  double Ca350 =rsc*integ3D(intMea,T350 );
  double Cb050 =rsc*integ3D(intMeb,T050 ); double Cb150 =rsc*integ3D(intMeb,T150 );double Cb250 =rsc*integ3D(intMeb,T250 );  double Cb350 =rsc*integ3D(intMeb,T350 );
  double Ca05p1=rsc*integ3D(intMea,T05p1);double Ca15p1=rsc*integ3D(intMea,T15p1);double Ca25p1=rsc*integ3D(intMea,T25p1);  double Ca35p1=rsc*integ3D(intMea,T35p1);
  double Cb05p1=rsc*integ3D(intMeb,T05p1);double Cb15p1=rsc*integ3D(intMeb,T15p1);double Cb25p1=rsc*integ3D(intMeb,T25p1);  double Cb35p1=rsc*integ3D(intMeb,T35p1);
  double Ca05p2=rsc*integ3D(intMea,T05p2);double Ca15p2=rsc*integ3D(intMea,T15p2);double Ca25p2=rsc*integ3D(intMea,T25p2);  double Ca35p2=rsc*integ3D(intMea,T35p2);
  double Cb05p2=rsc*integ3D(intMeb,T05p2);double Cb15p2=rsc*integ3D(intMeb,T15p2);double Cb25p2=rsc*integ3D(intMeb,T25p2);  double Cb35p2=rsc*integ3D(intMeb,T35p2);
  double Ca05p3=rsc*integ3D(intMea,T05p3);double Ca15p3=rsc*integ3D(intMea,T15p3);double Ca25p3=rsc*integ3D(intMea,T25p3);  double Ca35p3=rsc*integ3D(intMea,T35p3);
  double Cb05p3=rsc*integ3D(intMeb,T05p3);double Cb15p3=rsc*integ3D(intMeb,T15p3);double Cb25p3=rsc*integ3D(intMeb,T25p3);  double Cb35p3=rsc*integ3D(intMeb,T35p3);
  double Ca05p4=rsc*integ3D(intMea,T05p4);double Ca15p4=rsc*integ3D(intMea,T15p4);double Ca25p4=rsc*integ3D(intMea,T25p4);  double Ca35p4=rsc*integ3D(intMea,T35p4);
  double Cb05p4=rsc*integ3D(intMeb,T05p4);double Cb15p4=rsc*integ3D(intMeb,T15p4);double Cb25p4=rsc*integ3D(intMeb,T25p4);  double Cb35p4=rsc*integ3D(intMeb,T35p4);
  double Ca05p5=rsc*integ3D(intMea,T05p5);double Ca15p5=rsc*integ3D(intMea,T15p5);double Ca25p5=rsc*integ3D(intMea,T25p5);  double Ca35p5=rsc*integ3D(intMea,T35p5);
  double Cb05p5=rsc*integ3D(intMeb,T05p5);double Cb15p5=rsc*integ3D(intMeb,T15p5);double Cb25p5=rsc*integ3D(intMeb,T25p5);  double Cb35p5=rsc*integ3D(intMeb,T35p5);
  double Ca06m6=rsc*integ3D(intMea,T06m6);double Ca16m6=rsc*integ3D(intMea,T16m6);double Ca26m6=rsc*integ3D(intMea,T26m6);  double Ca36m6=rsc*integ3D(intMea,T36m6);
  double Cb06m6=rsc*integ3D(intMeb,T06m6);double Cb16m6=rsc*integ3D(intMeb,T16m6);double Cb26m6=rsc*integ3D(intMeb,T26m6);  double Cb36m6=rsc*integ3D(intMeb,T36m6);
  double Ca06m5=rsc*integ3D(intMea,T06m5);double Ca16m5=rsc*integ3D(intMea,T16m5);double Ca26m5=rsc*integ3D(intMea,T26m5);  double Ca36m5=rsc*integ3D(intMea,T36m5);
  double Cb06m5=rsc*integ3D(intMeb,T06m5);double Cb16m5=rsc*integ3D(intMeb,T16m5);double Cb26m5=rsc*integ3D(intMeb,T26m5);  double Cb36m5=rsc*integ3D(intMeb,T36m5);
  double Ca06m4=rsc*integ3D(intMea,T06m4);double Ca16m4=rsc*integ3D(intMea,T16m4);double Ca26m4=rsc*integ3D(intMea,T26m4);  double Ca36m4=rsc*integ3D(intMea,T36m4);
  double Cb06m4=rsc*integ3D(intMeb,T06m4);double Cb16m4=rsc*integ3D(intMeb,T16m4);double Cb26m4=rsc*integ3D(intMeb,T26m4);  double Cb36m4=rsc*integ3D(intMeb,T36m4);
  double Ca06m3=rsc*integ3D(intMea,T06m3);double Ca16m3=rsc*integ3D(intMea,T16m3);double Ca26m3=rsc*integ3D(intMea,T26m3);  double Ca36m3=rsc*integ3D(intMea,T36m3);
  double Cb06m3=rsc*integ3D(intMeb,T06m3);double Cb16m3=rsc*integ3D(intMeb,T16m3);double Cb26m3=rsc*integ3D(intMeb,T26m3);  double Cb36m3=rsc*integ3D(intMeb,T36m3);
  double Ca06m2=rsc*integ3D(intMea,T06m2);double Ca16m2=rsc*integ3D(intMea,T16m2);double Ca26m2=rsc*integ3D(intMea,T26m2);  double Ca36m2=rsc*integ3D(intMea,T36m2);
  double Cb06m2=rsc*integ3D(intMeb,T06m2);double Cb16m2=rsc*integ3D(intMeb,T16m2);double Cb26m2=rsc*integ3D(intMeb,T26m2);  double Cb36m2=rsc*integ3D(intMeb,T36m2);
  double Ca06m1=rsc*integ3D(intMea,T06m1);double Ca16m1=rsc*integ3D(intMea,T16m1);double Ca26m1=rsc*integ3D(intMea,T26m1);  double Ca36m1=rsc*integ3D(intMea,T36m1);
  double Cb06m1=rsc*integ3D(intMeb,T06m1);double Cb16m1=rsc*integ3D(intMeb,T16m1);double Cb26m1=rsc*integ3D(intMeb,T26m1);  double Cb36m1=rsc*integ3D(intMeb,T36m1);
  double Ca060 =rsc*integ3D(intMea,T060 ); double Ca160 =rsc*integ3D(intMea,T160 );double Ca260 =rsc*integ3D(intMea,T260 );  double Ca360 =rsc*integ3D(intMea,T360 );
  double Cb060 =rsc*integ3D(intMeb,T060 ); double Cb160 =rsc*integ3D(intMeb,T160 );double Cb260 =rsc*integ3D(intMeb,T260 );  double Cb360 =rsc*integ3D(intMeb,T360 );
  double Ca06p1=rsc*integ3D(intMea,T06p1);double Ca16p1=rsc*integ3D(intMea,T16p1);double Ca26p1=rsc*integ3D(intMea,T26p1);  double Ca36p1=rsc*integ3D(intMea,T36p1);
  double Cb06p1=rsc*integ3D(intMeb,T06p1);double Cb16p1=rsc*integ3D(intMeb,T16p1);double Cb26p1=rsc*integ3D(intMeb,T26p1);  double Cb36p1=rsc*integ3D(intMeb,T36p1);
  double Ca06p2=rsc*integ3D(intMea,T06p2);double Ca16p2=rsc*integ3D(intMea,T16p2);double Ca26p2=rsc*integ3D(intMea,T26p2);  double Ca36p2=rsc*integ3D(intMea,T36p2);
  double Cb06p2=rsc*integ3D(intMeb,T06p2);double Cb16p2=rsc*integ3D(intMeb,T16p2);double Cb26p2=rsc*integ3D(intMeb,T26p2);  double Cb36p2=rsc*integ3D(intMeb,T36p2);
  double Ca06p3=rsc*integ3D(intMea,T06p3);double Ca16p3=rsc*integ3D(intMea,T16p3);double Ca26p3=rsc*integ3D(intMea,T26p3);  double Ca36p3=rsc*integ3D(intMea,T36p3);
  double Cb06p3=rsc*integ3D(intMeb,T06p3);double Cb16p3=rsc*integ3D(intMeb,T16p3);double Cb26p3=rsc*integ3D(intMeb,T26p3);  double Cb36p3=rsc*integ3D(intMeb,T36p3);
  double Ca06p4=rsc*integ3D(intMea,T06p4);double Ca16p4=rsc*integ3D(intMea,T16p4);double Ca26p4=rsc*integ3D(intMea,T26p4);  double Ca36p4=rsc*integ3D(intMea,T36p4);
  double Cb06p4=rsc*integ3D(intMeb,T06p4);double Cb16p4=rsc*integ3D(intMeb,T16p4);double Cb26p4=rsc*integ3D(intMeb,T26p4);  double Cb36p4=rsc*integ3D(intMeb,T36p4);
  double Ca06p5=rsc*integ3D(intMea,T06p5);double Ca16p5=rsc*integ3D(intMea,T16p5);double Ca26p5=rsc*integ3D(intMea,T26p5);  double Ca36p5=rsc*integ3D(intMea,T36p5);
  double Cb06p5=rsc*integ3D(intMeb,T06p5);double Cb16p5=rsc*integ3D(intMeb,T16p5);double Cb26p5=rsc*integ3D(intMeb,T26p5);  double Cb36p5=rsc*integ3D(intMeb,T36p5);
  double Ca06p6=rsc*integ3D(intMea,T06p6);double Ca16p6=rsc*integ3D(intMea,T16p6);double Ca26p6=rsc*integ3D(intMea,T26p6);  double Ca36p6=rsc*integ3D(intMea,T36p6);
  double Cb06p6=rsc*integ3D(intMeb,T06p6);double Cb16p6=rsc*integ3D(intMeb,T16p6);double Cb26p6=rsc*integ3D(intMeb,T26p6);  double Cb36p6=rsc*integ3D(intMeb,T36p6);
  double Ca07m7=rsc*integ3D(intMea,T07m7);double Ca17m7=rsc*integ3D(intMea,T17m7);double Ca27m7=rsc*integ3D(intMea,T27m7);  double Ca37m7=rsc*integ3D(intMea,T37m7);
  double Cb07m7=rsc*integ3D(intMeb,T07m7);double Cb17m7=rsc*integ3D(intMeb,T17m7);double Cb27m7=rsc*integ3D(intMeb,T27m7);  double Cb37m7=rsc*integ3D(intMeb,T37m7);
  double Ca07m6=rsc*integ3D(intMea,T07m6);double Ca17m6=rsc*integ3D(intMea,T17m6);double Ca27m6=rsc*integ3D(intMea,T27m6);  double Ca37m6=rsc*integ3D(intMea,T37m6);
  double Cb07m6=rsc*integ3D(intMeb,T07m6);double Cb17m6=rsc*integ3D(intMeb,T17m6);double Cb27m6=rsc*integ3D(intMeb,T27m6);  double Cb37m6=rsc*integ3D(intMeb,T37m6);
  double Ca07m5=rsc*integ3D(intMea,T07m5);double Ca17m5=rsc*integ3D(intMea,T17m5);double Ca27m5=rsc*integ3D(intMea,T27m5);  double Ca37m5=rsc*integ3D(intMea,T37m5);
  double Cb07m5=rsc*integ3D(intMeb,T07m5);double Cb17m5=rsc*integ3D(intMeb,T17m5);double Cb27m5=rsc*integ3D(intMeb,T27m5);  double Cb37m5=rsc*integ3D(intMeb,T37m5);
  double Ca07m4=rsc*integ3D(intMea,T07m4);double Ca17m4=rsc*integ3D(intMea,T17m4);double Ca27m4=rsc*integ3D(intMea,T27m4);  double Ca37m4=rsc*integ3D(intMea,T37m4);
  double Cb07m4=rsc*integ3D(intMeb,T07m4);double Cb17m4=rsc*integ3D(intMeb,T17m4);double Cb27m4=rsc*integ3D(intMeb,T27m4);  double Cb37m4=rsc*integ3D(intMeb,T37m4);
  double Ca07m3=rsc*integ3D(intMea,T07m3);double Ca17m3=rsc*integ3D(intMea,T17m3);double Ca27m3=rsc*integ3D(intMea,T27m3);  double Ca37m3=rsc*integ3D(intMea,T37m3);
  double Cb07m3=rsc*integ3D(intMeb,T07m3);double Cb17m3=rsc*integ3D(intMeb,T17m3);double Cb27m3=rsc*integ3D(intMeb,T27m3);  double Cb37m3=rsc*integ3D(intMeb,T37m3);
  double Ca07m2=rsc*integ3D(intMea,T07m2);double Ca17m2=rsc*integ3D(intMea,T17m2);double Ca27m2=rsc*integ3D(intMea,T27m2);  double Ca37m2=rsc*integ3D(intMea,T37m2);
  double Cb07m2=rsc*integ3D(intMeb,T07m2);double Cb17m2=rsc*integ3D(intMeb,T17m2);double Cb27m2=rsc*integ3D(intMeb,T27m2);  double Cb37m2=rsc*integ3D(intMeb,T37m2);
  double Ca07m1=rsc*integ3D(intMea,T07m1);double Ca17m1=rsc*integ3D(intMea,T17m1);double Ca27m1=rsc*integ3D(intMea,T27m1);  double Ca37m1=rsc*integ3D(intMea,T37m1);
  double Cb07m1=rsc*integ3D(intMeb,T07m1);double Cb17m1=rsc*integ3D(intMeb,T17m1);double Cb27m1=rsc*integ3D(intMeb,T27m1);  double Cb37m1=rsc*integ3D(intMeb,T37m1);
  double Ca070 =rsc*integ3D(intMea,T070 ); double Ca170 =rsc*integ3D(intMea,T170 );double Ca270 =rsc*integ3D(intMea,T270 );  double Ca370 =rsc*integ3D(intMea,T370 );
  double Cb070 =rsc*integ3D(intMeb,T070 ); double Cb170 =rsc*integ3D(intMeb,T170 );double Cb270 =rsc*integ3D(intMeb,T270 );  double Cb370 =rsc*integ3D(intMeb,T370 );
  double Ca07p1=rsc*integ3D(intMea,T07p1);double Ca17p1=rsc*integ3D(intMea,T17p1);double Ca27p1=rsc*integ3D(intMea,T27p1);  double Ca37p1=rsc*integ3D(intMea,T37p1);
  double Cb07p1=rsc*integ3D(intMeb,T07p1);double Cb17p1=rsc*integ3D(intMeb,T17p1);double Cb27p1=rsc*integ3D(intMeb,T27p1);  double Cb37p1=rsc*integ3D(intMeb,T37p1);
  double Ca07p2=rsc*integ3D(intMea,T07p2);double Ca17p2=rsc*integ3D(intMea,T17p2);double Ca27p2=rsc*integ3D(intMea,T27p2);  double Ca37p2=rsc*integ3D(intMea,T37p2);
  double Cb07p2=rsc*integ3D(intMeb,T07p2);double Cb17p2=rsc*integ3D(intMeb,T17p2);double Cb27p2=rsc*integ3D(intMeb,T27p2);  double Cb37p2=rsc*integ3D(intMeb,T37p2);
  double Ca07p3=rsc*integ3D(intMea,T07p3);double Ca17p3=rsc*integ3D(intMea,T17p3);double Ca27p3=rsc*integ3D(intMea,T27p3);  double Ca37p3=rsc*integ3D(intMea,T37p3);
  double Cb07p3=rsc*integ3D(intMeb,T07p3);double Cb17p3=rsc*integ3D(intMeb,T17p3);double Cb27p3=rsc*integ3D(intMeb,T27p3);  double Cb37p3=rsc*integ3D(intMeb,T37p3);
  double Ca07p4=rsc*integ3D(intMea,T07p4);double Ca17p4=rsc*integ3D(intMea,T17p4);double Ca27p4=rsc*integ3D(intMea,T27p4);  double Ca37p4=rsc*integ3D(intMea,T37p4);
  double Cb07p4=rsc*integ3D(intMeb,T07p4);double Cb17p4=rsc*integ3D(intMeb,T17p4);double Cb27p4=rsc*integ3D(intMeb,T27p4);  double Cb37p4=rsc*integ3D(intMeb,T37p4);
  double Ca07p5=rsc*integ3D(intMea,T07p5);double Ca17p5=rsc*integ3D(intMea,T17p5);double Ca27p5=rsc*integ3D(intMea,T27p5);  double Ca37p5=rsc*integ3D(intMea,T37p5);
  double Cb07p5=rsc*integ3D(intMeb,T07p5);double Cb17p5=rsc*integ3D(intMeb,T17p5);double Cb27p5=rsc*integ3D(intMeb,T27p5);  double Cb37p5=rsc*integ3D(intMeb,T37p5);
  double Ca07p6=rsc*integ3D(intMea,T07p6);double Ca17p6=rsc*integ3D(intMea,T17p6);double Ca27p6=rsc*integ3D(intMea,T27p6);  double Ca37p6=rsc*integ3D(intMea,T37p6);
  double Cb07p6=rsc*integ3D(intMeb,T07p6);double Cb17p6=rsc*integ3D(intMeb,T17p6);double Cb27p6=rsc*integ3D(intMeb,T27p6);  double Cb37p6=rsc*integ3D(intMeb,T37p6);
  double Ca07p7=rsc*integ3D(intMea,T07p7);double Ca17p7=rsc*integ3D(intMea,T17p7);double Ca27p7=rsc*integ3D(intMea,T27p7);  double Ca37p7=rsc*integ3D(intMea,T37p7);
  double Cb07p7=rsc*integ3D(intMeb,T07p7);double Cb17p7=rsc*integ3D(intMeb,T17p7);double Cb27p7=rsc*integ3D(intMeb,T27p7);  double Cb37p7=rsc*integ3D(intMeb,T37p7);
  double Ca08m8=rsc*integ3D(intMea,T08m8); double Ca18m8=rsc*integ3D(intMea,T18m8); double Ca28m8=rsc*integ3D(intMea,T28m8);  double Ca38m8=rsc*integ3D(intMea,T38m8);
  double Cb08m8=rsc*integ3D(intMeb,T08m8); double Cb18m8=rsc*integ3D(intMeb,T18m8); double Cb28m8=rsc*integ3D(intMeb,T28m8);  double Cb38m8=rsc*integ3D(intMeb,T38m8);
  double Ca08m7=rsc*integ3D(intMea,T08m7); double Ca18m7=rsc*integ3D(intMea,T18m7); double Ca28m7=rsc*integ3D(intMea,T28m7);  double Ca38m7=rsc*integ3D(intMea,T38m7);
  double Cb08m7=rsc*integ3D(intMeb,T08m7); double Cb18m7=rsc*integ3D(intMeb,T18m7); double Cb28m7=rsc*integ3D(intMeb,T28m7);  double Cb38m7=rsc*integ3D(intMeb,T38m7);
  double Ca08m6=rsc*integ3D(intMea,T08m6); double Ca18m6=rsc*integ3D(intMea,T18m6); double Ca28m6=rsc*integ3D(intMea,T28m6);  double Ca38m6=rsc*integ3D(intMea,T38m6);
  double Cb08m6=rsc*integ3D(intMeb,T08m6); double Cb18m6=rsc*integ3D(intMeb,T18m6); double Cb28m6=rsc*integ3D(intMeb,T28m6);  double Cb38m6=rsc*integ3D(intMeb,T38m6);
  double Ca08m5=rsc*integ3D(intMea,T08m5); double Ca18m5=rsc*integ3D(intMea,T18m5); double Ca28m5=rsc*integ3D(intMea,T28m5);  double Ca38m5=rsc*integ3D(intMea,T38m5);
  double Cb08m5=rsc*integ3D(intMeb,T08m5); double Cb18m5=rsc*integ3D(intMeb,T18m5); double Cb28m5=rsc*integ3D(intMeb,T28m5);  double Cb38m5=rsc*integ3D(intMeb,T38m5);
  double Ca08m4=rsc*integ3D(intMea,T08m4); double Ca18m4=rsc*integ3D(intMea,T18m4); double Ca28m4=rsc*integ3D(intMea,T28m4);  double Ca38m4=rsc*integ3D(intMea,T38m4);
  double Cb08m4=rsc*integ3D(intMeb,T08m4); double Cb18m4=rsc*integ3D(intMeb,T18m4); double Cb28m4=rsc*integ3D(intMeb,T28m4);  double Cb38m4=rsc*integ3D(intMeb,T38m4);
  double Ca08m3=rsc*integ3D(intMea,T08m3); double Ca18m3=rsc*integ3D(intMea,T18m3); double Ca28m3=rsc*integ3D(intMea,T28m3);  double Ca38m3=rsc*integ3D(intMea,T38m3);
  double Cb08m3=rsc*integ3D(intMeb,T08m3); double Cb18m3=rsc*integ3D(intMeb,T18m3); double Cb28m3=rsc*integ3D(intMeb,T28m3);  double Cb38m3=rsc*integ3D(intMeb,T38m3);
  double Ca08m2=rsc*integ3D(intMea,T08m2); double Ca18m2=rsc*integ3D(intMea,T18m2); double Ca28m2=rsc*integ3D(intMea,T28m2);  double Ca38m2=rsc*integ3D(intMea,T38m2);
  double Cb08m2=rsc*integ3D(intMeb,T08m2); double Cb18m2=rsc*integ3D(intMeb,T18m2); double Cb28m2=rsc*integ3D(intMeb,T28m2);  double Cb38m2=rsc*integ3D(intMeb,T38m2);
  double Ca08m1=rsc*integ3D(intMea,T08m1); double Ca18m1=rsc*integ3D(intMea,T18m1); double Ca28m1=rsc*integ3D(intMea,T28m1);  double Ca38m1=rsc*integ3D(intMea,T38m1);
  double Cb08m1=rsc*integ3D(intMeb,T08m1); double Cb18m1=rsc*integ3D(intMeb,T18m1); double Cb28m1=rsc*integ3D(intMeb,T28m1);  double Cb38m1=rsc*integ3D(intMeb,T38m1);
  double Ca080 =rsc*integ3D(intMea,T080 );  double Ca180 =rsc*integ3D(intMea,T180 ); double Ca280 =rsc*integ3D(intMea,T280 );  double Ca380 =rsc*integ3D(intMea,T380 );
  double Cb080 =rsc*integ3D(intMeb,T080 );  double Cb180 =rsc*integ3D(intMeb,T180 ); double Cb280 =rsc*integ3D(intMeb,T280 );  double Cb380 =rsc*integ3D(intMeb,T380 );
  double Ca08p1=rsc*integ3D(intMea,T08p1); double Ca18p1=rsc*integ3D(intMea,T18p1); double Ca28p1=rsc*integ3D(intMea,T28p1);  double Ca38p1=rsc*integ3D(intMea,T38p1);
  double Cb08p1=rsc*integ3D(intMeb,T08p1); double Cb18p1=rsc*integ3D(intMeb,T18p1); double Cb28p1=rsc*integ3D(intMeb,T28p1);  double Cb38p1=rsc*integ3D(intMeb,T38p1);
  double Ca08p2=rsc*integ3D(intMea,T08p2); double Ca18p2=rsc*integ3D(intMea,T18p2); double Ca28p2=rsc*integ3D(intMea,T28p2);  double Ca38p2=rsc*integ3D(intMea,T38p2);
  double Cb08p2=rsc*integ3D(intMeb,T08p2); double Cb18p2=rsc*integ3D(intMeb,T18p2); double Cb28p2=rsc*integ3D(intMeb,T28p2);  double Cb38p2=rsc*integ3D(intMeb,T38p2);
  double Ca08p3=rsc*integ3D(intMea,T08p3); double Ca18p3=rsc*integ3D(intMea,T18p3); double Ca28p3=rsc*integ3D(intMea,T28p3);  double Ca38p3=rsc*integ3D(intMea,T38p3);
  double Cb08p3=rsc*integ3D(intMeb,T08p3); double Cb18p3=rsc*integ3D(intMeb,T18p3); double Cb28p3=rsc*integ3D(intMeb,T28p3);  double Cb38p3=rsc*integ3D(intMeb,T38p3);
  double Ca08p4=rsc*integ3D(intMea,T08p4); double Ca18p4=rsc*integ3D(intMea,T18p4); double Ca28p4=rsc*integ3D(intMea,T28p4);  double Ca38p4=rsc*integ3D(intMea,T38p4);
  double Cb08p4=rsc*integ3D(intMeb,T08p4); double Cb18p4=rsc*integ3D(intMeb,T18p4); double Cb28p4=rsc*integ3D(intMeb,T28p4);  double Cb38p4=rsc*integ3D(intMeb,T38p4);
  double Ca08p5=rsc*integ3D(intMea,T08p5); double Ca18p5=rsc*integ3D(intMea,T18p5); double Ca28p5=rsc*integ3D(intMea,T28p5);  double Ca38p5=rsc*integ3D(intMea,T38p5);
  double Cb08p5=rsc*integ3D(intMeb,T08p5); double Cb18p5=rsc*integ3D(intMeb,T18p5); double Cb28p5=rsc*integ3D(intMeb,T28p5);  double Cb38p5=rsc*integ3D(intMeb,T38p5);
  double Ca08p6=rsc*integ3D(intMea,T08p6); double Ca18p6=rsc*integ3D(intMea,T18p6); double Ca28p6=rsc*integ3D(intMea,T28p6);  double Ca38p6=rsc*integ3D(intMea,T38p6);
  double Cb08p6=rsc*integ3D(intMeb,T08p6); double Cb18p6=rsc*integ3D(intMeb,T18p6); double Cb28p6=rsc*integ3D(intMeb,T28p6);  double Cb38p6=rsc*integ3D(intMeb,T38p6);
  double Ca08p7=rsc*integ3D(intMea,T08p7); double Ca18p7=rsc*integ3D(intMea,T18p7); double Ca28p7=rsc*integ3D(intMea,T28p7);  double Ca38p7=rsc*integ3D(intMea,T38p7);
  double Cb08p7=rsc*integ3D(intMeb,T08p7); double Cb18p7=rsc*integ3D(intMeb,T18p7); double Cb28p7=rsc*integ3D(intMeb,T28p7);  double Cb38p7=rsc*integ3D(intMeb,T38p7);
  double Ca08p8=rsc*integ3D(intMea,T08p8); double Ca18p8=rsc*integ3D(intMea,T18p8); double Ca28p8=rsc*integ3D(intMea,T28p8);  double Ca38p8=rsc*integ3D(intMea,T38p8);
  double Cb08p8=rsc*integ3D(intMeb,T08p8); double Cb18p8=rsc*integ3D(intMeb,T18p8); double Cb28p8=rsc*integ3D(intMeb,T28p8);  double Cb38p8=rsc*integ3D(intMeb,T38p8);
  double Ca09m9=rsc*integ3D(intMea,T09m9); double Ca19m9=rsc*integ3D(intMea,T19m9); double Ca29m9=rsc*integ3D(intMea,T29m9);  double Ca39m9=rsc*integ3D(intMea,T39m9);
  double Cb09m9=rsc*integ3D(intMeb,T09m9); double Cb19m9=rsc*integ3D(intMeb,T19m9); double Cb29m9=rsc*integ3D(intMeb,T29m9);  double Cb39m9=rsc*integ3D(intMeb,T39m9);
  double Ca09m8=rsc*integ3D(intMea,T09m8); double Ca19m8=rsc*integ3D(intMea,T19m8); double Ca29m8=rsc*integ3D(intMea,T29m8);  double Ca39m8=rsc*integ3D(intMea,T39m8);
  double Cb09m8=rsc*integ3D(intMeb,T09m8); double Cb19m8=rsc*integ3D(intMeb,T19m8); double Cb29m8=rsc*integ3D(intMeb,T29m8);  double Cb39m8=rsc*integ3D(intMeb,T39m8);
  double Ca09m7=rsc*integ3D(intMea,T09m7); double Ca19m7=rsc*integ3D(intMea,T19m7); double Ca29m7=rsc*integ3D(intMea,T29m7);  double Ca39m7=rsc*integ3D(intMea,T39m7);
  double Cb09m7=rsc*integ3D(intMeb,T09m7); double Cb19m7=rsc*integ3D(intMeb,T19m7); double Cb29m7=rsc*integ3D(intMeb,T29m7);  double Cb39m7=rsc*integ3D(intMeb,T39m7);
  double Ca09m6=rsc*integ3D(intMea,T09m6); double Ca19m6=rsc*integ3D(intMea,T19m6); double Ca29m6=rsc*integ3D(intMea,T29m6);  double Ca39m6=rsc*integ3D(intMea,T39m6);
  double Cb09m6=rsc*integ3D(intMeb,T09m6); double Cb19m6=rsc*integ3D(intMeb,T19m6); double Cb29m6=rsc*integ3D(intMeb,T29m6);  double Cb39m6=rsc*integ3D(intMeb,T39m6);
  double Ca09m5=rsc*integ3D(intMea,T09m5); double Ca19m5=rsc*integ3D(intMea,T19m5); double Ca29m5=rsc*integ3D(intMea,T29m5);  double Ca39m5=rsc*integ3D(intMea,T39m5);
  double Cb09m5=rsc*integ3D(intMeb,T09m5); double Cb19m5=rsc*integ3D(intMeb,T19m5); double Cb29m5=rsc*integ3D(intMeb,T29m5);  double Cb39m5=rsc*integ3D(intMeb,T39m5);
  double Ca09m4=rsc*integ3D(intMea,T09m4); double Ca19m4=rsc*integ3D(intMea,T19m4); double Ca29m4=rsc*integ3D(intMea,T29m4);  double Ca39m4=rsc*integ3D(intMea,T39m4);
  double Cb09m4=rsc*integ3D(intMeb,T09m4); double Cb19m4=rsc*integ3D(intMeb,T19m4); double Cb29m4=rsc*integ3D(intMeb,T29m4);  double Cb39m4=rsc*integ3D(intMeb,T39m4);
  double Ca09m3=rsc*integ3D(intMea,T09m3); double Ca19m3=rsc*integ3D(intMea,T19m3); double Ca29m3=rsc*integ3D(intMea,T29m3);  double Ca39m3=rsc*integ3D(intMea,T39m3);
  double Cb09m3=rsc*integ3D(intMeb,T09m3); double Cb19m3=rsc*integ3D(intMeb,T19m3); double Cb29m3=rsc*integ3D(intMeb,T29m3);  double Cb39m3=rsc*integ3D(intMeb,T39m3);
  double Ca09m2=rsc*integ3D(intMea,T09m2); double Ca19m2=rsc*integ3D(intMea,T19m2); double Ca29m2=rsc*integ3D(intMea,T29m2);  double Ca39m2=rsc*integ3D(intMea,T39m2);
  double Cb09m2=rsc*integ3D(intMeb,T09m2); double Cb19m2=rsc*integ3D(intMeb,T19m2); double Cb29m2=rsc*integ3D(intMeb,T29m2);  double Cb39m2=rsc*integ3D(intMeb,T39m2);
  double Ca09m1=rsc*integ3D(intMea,T09m1); double Ca19m1=rsc*integ3D(intMea,T19m1); double Ca29m1=rsc*integ3D(intMea,T29m1);  double Ca39m1=rsc*integ3D(intMea,T39m1);
  double Cb09m1=rsc*integ3D(intMeb,T09m1); double Cb19m1=rsc*integ3D(intMeb,T19m1); double Cb29m1=rsc*integ3D(intMeb,T29m1);  double Cb39m1=rsc*integ3D(intMeb,T39m1);
  double Ca090 =rsc*integ3D(intMea,T090 );  double Ca190 =rsc*integ3D(intMea,T190 ); double Ca290 =rsc*integ3D(intMea,T290 );  double Ca390 =rsc*integ3D(intMea,T390 );
  double Cb090 =rsc*integ3D(intMeb,T090 );  double Cb190 =rsc*integ3D(intMeb,T190 ); double Cb290 =rsc*integ3D(intMeb,T290 );  double Cb390 =rsc*integ3D(intMeb,T390 );
  double Ca09p1=rsc*integ3D(intMea,T09p1); double Ca19p1=rsc*integ3D(intMea,T19p1); double Ca29p1=rsc*integ3D(intMea,T29p1);  double Ca39p1=rsc*integ3D(intMea,T39p1);
  double Cb09p1=rsc*integ3D(intMeb,T09p1); double Cb19p1=rsc*integ3D(intMeb,T19p1); double Cb29p1=rsc*integ3D(intMeb,T29p1);  double Cb39p1=rsc*integ3D(intMeb,T39p1);
  double Ca09p2=rsc*integ3D(intMea,T09p2); double Ca19p2=rsc*integ3D(intMea,T19p2); double Ca29p2=rsc*integ3D(intMea,T29p2);  double Ca39p2=rsc*integ3D(intMea,T39p2);
  double Cb09p2=rsc*integ3D(intMeb,T09p2); double Cb19p2=rsc*integ3D(intMeb,T19p2); double Cb29p2=rsc*integ3D(intMeb,T29p2);  double Cb39p2=rsc*integ3D(intMeb,T39p2);
  double Ca09p3=rsc*integ3D(intMea,T09p3); double Ca19p3=rsc*integ3D(intMea,T19p3); double Ca29p3=rsc*integ3D(intMea,T29p3);  double Ca39p3=rsc*integ3D(intMea,T39p3);
  double Cb09p3=rsc*integ3D(intMeb,T09p3); double Cb19p3=rsc*integ3D(intMeb,T19p3); double Cb29p3=rsc*integ3D(intMeb,T29p3);  double Cb39p3=rsc*integ3D(intMeb,T39p3);
  double Ca09p4=rsc*integ3D(intMea,T09p4); double Ca19p4=rsc*integ3D(intMea,T19p4); double Ca29p4=rsc*integ3D(intMea,T29p4);  double Ca39p4=rsc*integ3D(intMea,T39p4);
  double Cb09p4=rsc*integ3D(intMeb,T09p4); double Cb19p4=rsc*integ3D(intMeb,T19p4); double Cb29p4=rsc*integ3D(intMeb,T29p4);  double Cb39p4=rsc*integ3D(intMeb,T39p4);
  double Ca09p5=rsc*integ3D(intMea,T09p5); double Ca19p5=rsc*integ3D(intMea,T19p5); double Ca29p5=rsc*integ3D(intMea,T29p5);  double Ca39p5=rsc*integ3D(intMea,T39p5);
  double Cb09p5=rsc*integ3D(intMeb,T09p5); double Cb19p5=rsc*integ3D(intMeb,T19p5); double Cb29p5=rsc*integ3D(intMeb,T29p5);  double Cb39p5=rsc*integ3D(intMeb,T39p5);
  double Ca09p6=rsc*integ3D(intMea,T09p6); double Ca19p6=rsc*integ3D(intMea,T19p6); double Ca29p6=rsc*integ3D(intMea,T29p6);  double Ca39p6=rsc*integ3D(intMea,T39p6);
  double Cb09p6=rsc*integ3D(intMeb,T09p6); double Cb19p6=rsc*integ3D(intMeb,T19p6); double Cb29p6=rsc*integ3D(intMeb,T29p6);  double Cb39p6=rsc*integ3D(intMeb,T39p6);
  double Ca09p7=rsc*integ3D(intMea,T09p7); double Ca19p7=rsc*integ3D(intMea,T19p7); double Ca29p7=rsc*integ3D(intMea,T29p7);  double Ca39p7=rsc*integ3D(intMea,T39p7);
  double Cb09p7=rsc*integ3D(intMeb,T09p7); double Cb19p7=rsc*integ3D(intMeb,T19p7); double Cb29p7=rsc*integ3D(intMeb,T29p7);  double Cb39p7=rsc*integ3D(intMeb,T39p7);
  double Ca09p8=rsc*integ3D(intMea,T09p8); double Ca19p8=rsc*integ3D(intMea,T19p8); double Ca29p8=rsc*integ3D(intMea,T29p8);  double Ca39p8=rsc*integ3D(intMea,T39p8);
  double Cb09p8=rsc*integ3D(intMeb,T09p8); double Cb19p8=rsc*integ3D(intMeb,T19p8); double Cb29p8=rsc*integ3D(intMeb,T29p8);  double Cb39p8=rsc*integ3D(intMeb,T39p8);
  double Ca09p9=rsc*integ3D(intMea,T09p9); double Ca19p9=rsc*integ3D(intMea,T19p9); double Ca29p9=rsc*integ3D(intMea,T29p9);  double Ca39p9=rsc*integ3D(intMea,T39p9);
  double Cb09p9=rsc*integ3D(intMeb,T09p9); double Cb19p9=rsc*integ3D(intMeb,T19p9); double Cb29p9=rsc*integ3D(intMeb,T29p9);  double Cb39p9=rsc*integ3D(intMeb,T39p9);

//double C[3][10][21];
//memset(C, 0.0, sizeof C);
  cube C = zeros<cube>(2,4,100);
    
 // Saving the values in C[n][l]. 
  C(0,0,0) = Ca000 ; C(0,1,0) = Ca100; C(0,2,0) =  Ca200;  C(0,3,0) = Ca300 ;
  C(1,0,0) = Cb000 ; C(1,1,0) = Cb100; C(1,2,0) =  Cb200;  C(1,3,0) = Cb300 ;
  C(0,0,1) = Ca01m1 ; C(0,1,1) = Ca11m1; C(0,2,1) =  Ca21m1;  C(0,3,1) = Ca31m1 ;
  C(1,0,1) = Cb01m1 ; C(1,1,1) = Cb11m1; C(1,2,1) =  Cb21m1;  C(1,3,1) = Cb31m1 ;
  C(0,0,2) = Ca010  ; C(0,1,2) = Ca110 ; C(0,2,2) =  Ca210 ;  C(0,3,2) = Ca310  ;
  C(1,0,2) = Cb010  ; C(1,1,2) = Cb110 ; C(1,2,2) =  Cb210 ;  C(1,3,2) = Cb310  ;
  C(0,0,3) = Ca01p1 ; C(0,1,3) = Ca11p1; C(0,2,3) =  Ca21p1;  C(0,3,3) = Ca31p1 ;
  C(1,0,3) = Cb01p1 ; C(1,1,3) = Cb11p1; C(1,2,3) =  Cb21p1;  C(1,3,3) = Cb31p1 ;
  C(0,0,4) = Ca02m2 ; C(0,1,4) = Ca12m2; C(0,2,4) =  Ca22m2;  C(0,3,4) = Ca32m2 ;
  C(1,0,4) = Cb02m2 ; C(1,1,4) = Cb12m2; C(1,2,4) =  Cb22m2;  C(1,3,4) = Cb32m2 ;
  C(0,0,5) = Ca02m1 ; C(0,1,5) = Ca12m1; C(0,2,5) =  Ca22m1;  C(0,3,5) = Ca32m1 ;
  C(1,0,5) = Cb02m1 ; C(1,1,5) = Cb12m1; C(1,2,5) =  Cb22m1;  C(1,3,5) = Cb32m1 ;
  C(0,0,6) = Ca020  ; C(0,1,6) = Ca120 ; C(0,2,6) =  Ca220 ;  C(0,3,6) = Ca320  ;
  C(1,0,6) = Cb020  ; C(1,1,6) = Cb120 ; C(1,2,6) =  Cb220 ;  C(1,3,6) = Cb320  ;
  C(0,0,7) = Ca02p1 ; C(0,1,7) = Ca12p1; C(0,2,7) =  Ca22p1;  C(0,3,7) = Ca32p1 ;
  C(1,0,7) = Cb02p1 ; C(1,1,7) = Cb12p1; C(1,2,7) =  Cb22p1;  C(1,3,7) = Cb32p1 ;
  C(0,0,8) = Ca02p2 ; C(0,1,8) = Ca12p2; C(0,2,8) =  Ca22p2;  C(0,3,8) = Ca32p2 ;
  C(1,0,8) = Cb02p2 ; C(1,1,8) = Cb12p2; C(1,2,8) =  Cb22p2;  C(1,3,8) = Cb32p2 ;
  C(0,0,9) = Ca03m3 ; C(0,1,9) = Ca13m3; C(0,2,9) =  Ca23m3;  C(0,3,9) = Ca33m3 ;
  C(1,0,9) = Cb03m3 ; C(1,1,9) = Cb13m3; C(1,2,9) =  Cb23m3;  C(1,3,9) = Cb33m3 ;
  C(0,0,10) = Ca03m2 ; C(0,1,10) = Ca13m2; C(0,2,10) =  Ca23m2;  C(0,3,10) = Ca33m2 ;
  C(1,0,10) = Cb03m2 ; C(1,1,10) = Cb13m2; C(1,2,10) =  Cb23m2;  C(1,3,10) = Cb33m2 ;
  C(0,0,11) = Ca03m1 ; C(0,1,11) = Ca13m1; C(0,2,11) =  Ca23m1;  C(0,3,11) = Ca33m1 ;
  C(1,0,11) = Cb03m1 ; C(1,1,11) = Cb13m1; C(1,2,11) =  Cb23m1;  C(1,3,11) = Cb33m1 ;
  C(0,0,12) = Ca030  ; C(0,1,12) = Ca130 ; C(0,2,12) =  Ca230 ;  C(0,3,12) = Ca330  ;
  C(1,0,12) = Cb030  ; C(1,1,12) = Cb130 ; C(1,2,12) =  Cb230 ;  C(1,3,12) = Cb330  ;
  C(0,0,13) = Ca03p1 ; C(0,1,13) = Ca13p1; C(0,2,13) =  Ca23p1;  C(0,3,13) = Ca33p1 ;
  C(1,0,13) = Cb03p1 ; C(1,1,13) = Cb13p1; C(1,2,13) =  Cb23p1;  C(1,3,13) = Cb33p1 ;
  C(0,0,14) = Ca03p2 ; C(0,1,14) = Ca13p2; C(0,2,14) =  Ca23p2;  C(0,3,14) = Ca33p2 ;
  C(1,0,14) = Cb03p2 ; C(1,1,14) = Cb13p2; C(1,2,14) =  Cb23p2;  C(1,3,14) = Cb33p2 ;
  C(0,0,15) = Ca03p3 ; C(0,1,15) = Ca13p3; C(0,2,15) =  Ca23p3;  C(0,3,15) = Ca33p3 ;
  C(1,0,15) = Cb03p3 ; C(1,1,15) = Cb13p3; C(1,2,15) =  Cb23p3;  C(1,3,15) = Cb33p3 ;
  C(0,0,16) = Ca04m4 ; C(0,1,16) = Ca14m4; C(0,2,16) =  Ca24m4;  C(0,3,16) = Ca34m4 ;
  C(1,0,16) = Cb04m4 ; C(1,1,16) = Cb14m4; C(1,2,16) =  Cb24m4;  C(1,3,16) = Cb34m4 ;
  C(0,0,17) = Ca04m3 ; C(0,1,17) = Ca14m3; C(0,2,17) =  Ca24m3;  C(0,3,17) = Ca34m3 ;
  C(1,0,17) = Cb04m3 ; C(1,1,17) = Cb14m3; C(1,2,17) =  Cb24m3;  C(1,3,17) = Cb34m3 ;
  C(0,0,18) = Ca04m2 ; C(0,1,18) = Ca14m2; C(0,2,18) =  Ca24m2;  C(0,3,18) = Ca34m2 ;
  C(1,0,18) = Cb04m2 ; C(1,1,18) = Cb14m2; C(1,2,18) =  Cb24m2;  C(1,3,18) = Cb34m2 ;
  C(0,0,19) = Ca04m1 ; C(0,1,19) = Ca14m1; C(0,2,19) =  Ca24m1;  C(0,3,19) = Ca34m1 ;
  C(1,0,19) = Cb04m1 ; C(1,1,19) = Cb14m1; C(1,2,19) =  Cb24m1;  C(1,3,19) = Cb34m1 ;
  C(0,0,20) = Ca040  ; C(0,1,20) = Ca140 ; C(0,2,20) =  Ca240 ;  C(0,3,20) = Ca340  ;
  C(1,0,20) = Cb040  ; C(1,1,20) = Cb140 ; C(1,2,20) =  Cb240 ;  C(1,3,20) = Cb340  ;
  C(0,0,21) = Ca04p1 ; C(0,1,21) = Ca14p1; C(0,2,21) =  Ca24p1;  C(0,3,21) = Ca34p1 ;
  C(1,0,21) = Cb04p1 ; C(1,1,21) = Cb14p1; C(1,2,21) =  Cb24p1;  C(1,3,21) = Cb34p1 ;
  C(0,0,22) = Ca04p2 ; C(0,1,22) = Ca14p2; C(0,2,22) =  Ca24p2;  C(0,3,22) = Ca34p2 ;
  C(1,0,22) = Cb04p2 ; C(1,1,22) = Cb14p2; C(1,2,22) =  Cb24p2;  C(1,3,22) = Cb34p2 ;
  C(0,0,23) = Ca04p3 ; C(0,1,23) = Ca14p3; C(0,2,23) =  Ca24p3;  C(0,3,23) = Ca34p3 ;
  C(1,0,23) = Cb04p3 ; C(1,1,23) = Cb14p3; C(1,2,23) =  Cb24p3;  C(1,3,23) = Cb34p3 ;
  C(0,0,24) = Ca04p4 ; C(0,1,24) = Ca14p4; C(0,2,24) =  Ca24p4;  C(0,3,24) = Ca34p4 ;
  C(1,0,24) = Cb04p4 ; C(1,1,24) = Cb14p4; C(1,2,24) =  Cb24p4;  C(1,3,24) = Cb34p4 ;
  C(0,0,25) = Ca05m5 ; C(0,1,25) = Ca15m5; C(0,2,25) =  Ca25m5;  C(0,3,25) = Ca35m5 ;
  C(1,0,25) = Cb05m5 ; C(1,1,25) = Cb15m5; C(1,2,25) =  Cb25m5;  C(1,3,25) = Cb35m5 ;
  C(0,0,26) = Ca05m4 ; C(0,1,26) = Ca15m4; C(0,2,26) =  Ca25m4;  C(0,3,26) = Ca35m4 ;
  C(1,0,26) = Cb05m4 ; C(1,1,26) = Cb15m4; C(1,2,26) =  Cb25m4;  C(1,3,26) = Cb35m4 ;
  C(0,0,27) = Ca05m3 ; C(0,1,27) = Ca15m3; C(0,2,27) =  Ca25m3;  C(0,3,27) = Ca35m3 ;
  C(1,0,27) = Cb05m3 ; C(1,1,27) = Cb15m3; C(1,2,27) =  Cb25m3;  C(1,3,27) = Cb35m3 ;
  C(0,0,28) = Ca05m2 ; C(0,1,28) = Ca15m2; C(0,2,28) =  Ca25m2;  C(0,3,28) = Ca35m2 ;
  C(1,0,28) = Cb05m2 ; C(1,1,28) = Cb15m2; C(1,2,28) =  Cb25m2;  C(1,3,28) = Cb35m2 ;
  C(0,0,29) = Ca05m1 ; C(0,1,29) = Ca15m1; C(0,2,29) =  Ca25m1;  C(0,3,29) = Ca35m1 ;
  C(1,0,29) = Cb05m1 ; C(1,1,29) = Cb15m1; C(1,2,29) =  Cb25m1;  C(1,3,29) = Cb35m1 ;
  C(0,0,30) = Ca050  ; C(0,1,30) = Ca150 ; C(0,2,30) =  Ca250 ;  C(0,3,30) = Ca350  ;
  C(1,0,30) = Cb050  ; C(1,1,30) = Cb150 ; C(1,2,30) =  Cb250 ;  C(1,3,30) = Cb350  ;
  C(0,0,31) = Ca05p1 ; C(0,1,31) = Ca15p1; C(0,2,31) =  Ca25p1;  C(0,3,31) = Ca35p1 ;
  C(1,0,31) = Cb05p1 ; C(1,1,31) = Cb15p1; C(1,2,31) =  Cb25p1;  C(1,3,31) = Cb35p1 ;
  C(0,0,32) = Ca05p2 ; C(0,1,32) = Ca15p2; C(0,2,32) =  Ca25p2;  C(0,3,32) = Ca35p2 ;
  C(1,0,32) = Cb05p2 ; C(1,1,32) = Cb15p2; C(1,2,32) =  Cb25p2;  C(1,3,32) = Cb35p2 ;
  C(0,0,33) = Ca05p3 ; C(0,1,33) = Ca15p3; C(0,2,33) =  Ca25p3;  C(0,3,33) = Ca35p3 ;
  C(1,0,33) = Cb05p3 ; C(1,1,33) = Cb15p3; C(1,2,33) =  Cb25p3;  C(1,3,33) = Cb35p3 ;
  C(0,0,34) = Ca05p4 ; C(0,1,34) = Ca15p4; C(0,2,34) =  Ca25p4;  C(0,3,34) = Ca35p4 ;
  C(1,0,34) = Cb05p4 ; C(1,1,34) = Cb15p4; C(1,2,34) =  Cb25p4;  C(1,3,34) = Cb35p4 ;
  C(0,0,35) = Ca05p5 ; C(0,1,35) = Ca15p5; C(0,2,35) =  Ca25p5;  C(0,3,35) = Ca35p5 ;
  C(1,0,35) = Cb05p5 ; C(1,1,35) = Cb15p5; C(1,2,35) =  Cb25p5;  C(1,3,35) = Cb35p5 ;
  C(0,0,36) = Ca06m6 ; C(0,1,36) = Ca16m6; C(0,2,36) =  Ca26m6;  C(0,3,36) = Ca36m6 ;
  C(1,0,36) = Cb06m6 ; C(1,1,36) = Cb16m6; C(1,2,36) =  Cb26m6;  C(1,3,36) = Cb36m6 ;
  C(0,0,37) = Ca06m5 ; C(0,1,37) = Ca16m5; C(0,2,37) =  Ca26m5;  C(0,3,37) = Ca36m5 ;
  C(1,0,37) = Cb06m5 ; C(1,1,37) = Cb16m5; C(1,2,37) =  Cb26m5;  C(1,3,37) = Cb36m5 ;
  C(0,0,38) = Ca06m4 ; C(0,1,38) = Ca16m4; C(0,2,38) =  Ca26m4;  C(0,3,38) = Ca36m4 ;
  C(1,0,38) = Cb06m4 ; C(1,1,38) = Cb16m4; C(1,2,38) =  Cb26m4;  C(1,3,38) = Cb36m4 ;
  C(0,0,39) = Ca06m3 ; C(0,1,39) = Ca16m3; C(0,2,39) =  Ca26m3;  C(0,3,39) = Ca36m3 ;
  C(1,0,39) = Cb06m3 ; C(1,1,39) = Cb16m3; C(1,2,39) =  Cb26m3;  C(1,3,39) = Cb36m3 ;
  C(0,0,40) = Ca06m2 ; C(0,1,40) = Ca16m2; C(0,2,40) =  Ca26m2;  C(0,3,40) = Ca36m2 ;
  C(1,0,40) = Cb06m2 ; C(1,1,40) = Cb16m2; C(1,2,40) =  Cb26m2;  C(1,3,40) = Cb36m2 ;
  C(0,0,41) = Ca06m1 ; C(0,1,41) = Ca16m1; C(0,2,41) =  Ca26m1;  C(0,3,41) = Ca36m1 ;
  C(1,0,41) = Cb06m1 ; C(1,1,41) = Cb16m1; C(1,2,41) =  Cb26m1;  C(1,3,41) = Cb36m1 ;
  C(0,0,42) = Ca060  ; C(0,1,42) = Ca160 ; C(0,2,42) =  Ca260 ;  C(0,3,42) = Ca360  ;
  C(1,0,42) = Cb060  ; C(1,1,42) = Cb160 ; C(1,2,42) =  Cb260 ;  C(1,3,42) = Cb360  ;
  C(0,0,43) = Ca06p1 ; C(0,1,43) = Ca16p1; C(0,2,43) =  Ca26p1;  C(0,3,43) = Ca36p1 ;
  C(1,0,43) = Cb06p1 ; C(1,1,43) = Cb16p1; C(1,2,43) =  Cb26p1;  C(1,3,43) = Cb36p1 ;
  C(0,0,44) = Ca06p2 ; C(0,1,44) = Ca16p2; C(0,2,44) =  Ca26p2;  C(0,3,44) = Ca36p2 ;
  C(1,0,44) = Cb06p2 ; C(1,1,44) = Cb16p2; C(1,2,44) =  Cb26p2;  C(1,3,44) = Cb36p2 ;
  C(0,0,45) = Ca06p3 ; C(0,1,45) = Ca16p3; C(0,2,45) =  Ca26p3;  C(0,3,45) = Ca36p3 ;
  C(1,0,45) = Cb06p3 ; C(1,1,45) = Cb16p3; C(1,2,45) =  Cb26p3;  C(1,3,45) = Cb36p3 ;
  C(0,0,46) = Ca06p4 ; C(0,1,46) = Ca16p4; C(0,2,46) =  Ca26p4;  C(0,3,46) = Ca36p4 ;
  C(1,0,46) = Cb06p4 ; C(1,1,46) = Cb16p4; C(1,2,46) =  Cb26p4;  C(1,3,46) = Cb36p4 ;
  C(0,0,47) = Ca06p5 ; C(0,1,47) = Ca16p5; C(0,2,47) =  Ca26p5;  C(0,3,47) = Ca36p5 ;
  C(1,0,47) = Cb06p5 ; C(1,1,47) = Cb16p5; C(1,2,47) =  Cb26p5;  C(1,3,47) = Cb36p5 ;
  C(0,0,48) = Ca06p6 ; C(0,1,48) = Ca16p6; C(0,2,48) =  Ca26p6;  C(0,3,48) = Ca36p6 ;
  C(1,0,48) = Cb06p6 ; C(1,1,48) = Cb16p6; C(1,2,48) =  Cb26p6;  C(1,3,48) = Cb36p6 ;
  C(0,0,49) = Ca07m7 ; C(0,1,49) = Ca17m7; C(0,2,49) =  Ca27m7;  C(0,3,49) = Ca37m7 ;
  C(1,0,49) = Cb07m7 ; C(1,1,49) = Cb17m7; C(1,2,49) =  Cb27m7;  C(1,3,49) = Cb37m7 ;
  C(0,0,50) = Ca07m6 ; C(0,1,50) = Ca17m6; C(0,2,50) =  Ca27m6;  C(0,3,50) = Ca37m6 ;
  C(1,0,50) = Cb07m6 ; C(1,1,50) = Cb17m6; C(1,2,50) =  Cb27m6;  C(1,3,50) = Cb37m6 ;
  C(0,0,51) = Ca07m5 ; C(0,1,51) = Ca17m5; C(0,2,51) =  Ca27m5;  C(0,3,51) = Ca37m5 ;
  C(1,0,51) = Cb07m5 ; C(1,1,51) = Cb17m5; C(1,2,51) =  Cb27m5;  C(1,3,51) = Cb37m5 ;
  C(0,0,52) = Ca07m4 ; C(0,1,52) = Ca17m4; C(0,2,52) =  Ca27m4;  C(0,3,52) = Ca37m4 ;
  C(1,0,52) = Cb07m4 ; C(1,1,52) = Cb17m4; C(1,2,52) =  Cb27m4;  C(1,3,52) = Cb37m4 ;
  C(0,0,53) = Ca07m3 ; C(0,1,53) = Ca17m3; C(0,2,53) =  Ca27m3;  C(0,3,53) = Ca37m3 ;
  C(1,0,53) = Cb07m3 ; C(1,1,53) = Cb17m3; C(1,2,53) =  Cb27m3;  C(1,3,53) = Cb37m3 ;
  C(0,0,54) = Ca07m2 ; C(0,1,54) = Ca17m2; C(0,2,54) =  Ca27m2;  C(0,3,54) = Ca37m2 ;
  C(1,0,54) = Cb07m2 ; C(1,1,54) = Cb17m2; C(1,2,54) =  Cb27m2;  C(1,3,54) = Cb37m2 ;
  C(0,0,55) = Ca07m1 ; C(0,1,55) = Ca17m1; C(0,2,55) =  Ca27m1;  C(0,3,55) = Ca37m1 ;
  C(1,0,55) = Cb07m1 ; C(1,1,55) = Cb17m1; C(1,2,55) =  Cb27m1;  C(1,3,55) = Cb37m1 ;
  C(0,0,56) = Ca070  ; C(0,1,56) = Ca170 ; C(0,2,56) =  Ca270 ;  C(0,3,56) = Ca370  ;
  C(1,0,56) = Cb070  ; C(1,1,56) = Cb170 ; C(1,2,56) =  Cb270 ;  C(1,3,56) = Cb370  ;
  C(0,0,57) = Ca07p1 ; C(0,1,57) = Ca17p1; C(0,2,57) =  Ca27p1;  C(0,3,57) = Ca37p1 ;
  C(1,0,57) = Cb07p1 ; C(1,1,57) = Cb17p1; C(1,2,57) =  Cb27p1;  C(1,3,57) = Cb37p1 ;
  C(0,0,58) = Ca07p2 ; C(0,1,58) = Ca17p2; C(0,2,58) =  Ca27p2;  C(0,3,58) = Ca37p2 ;
  C(1,0,58) = Cb07p2 ; C(1,1,58) = Cb17p2; C(1,2,58) =  Cb27p2;  C(1,3,58) = Cb37p2 ;
  C(0,0,59) = Ca07p3 ; C(0,1,59) = Ca17p3; C(0,2,59) =  Ca27p3;  C(0,3,59) = Ca37p3 ;
  C(1,0,59) = Cb07p3 ; C(1,1,59) = Cb17p3; C(1,2,59) =  Cb27p3;  C(1,3,59) = Cb37p3 ;
  C(0,0,60) = Ca07p4 ; C(0,1,60) = Ca17p4; C(0,2,60) =  Ca27p4;  C(0,3,60) = Ca37p4 ;
  C(1,0,60) = Cb07p4 ; C(1,1,60) = Cb17p4; C(1,2,60) =  Cb27p4;  C(1,3,60) = Cb37p4 ;
  C(0,0,61) = Ca07p5 ; C(0,1,61) = Ca17p5; C(0,2,61) =  Ca27p5;  C(0,3,61) = Ca37p5 ;
  C(1,0,61) = Cb07p5 ; C(1,1,61) = Cb17p5; C(1,2,61) =  Cb27p5;  C(1,3,61) = Cb37p5 ;
  C(0,0,62) = Ca07p6 ; C(0,1,62) = Ca17p6; C(0,2,62) =  Ca27p6;  C(0,3,62) = Ca37p6 ;
  C(1,0,62) = Cb07p6 ; C(1,1,62) = Cb17p6; C(1,2,62) =  Cb27p6;  C(1,3,62) = Cb37p6 ;
  C(0,0,63) = Ca07p7 ; C(0,1,63) = Ca17p7; C(0,2,63) =  Ca27p7;  C(0,3,63) = Ca37p7 ;
  C(1,0,63) = Cb07p7 ; C(1,1,63) = Cb17p7; C(1,2,63) =  Cb27p7;  C(1,3,63) = Cb37p7 ;
  C(0,0,64) = Ca08m8 ; C(0,1,64) = Ca18m8; C(0,2,64) =  Ca28m8;  C(0,3,64) = Ca38m8 ;
  C(1,0,64) = Cb08m8 ; C(1,1,64) = Cb18m8; C(1,2,64) =  Cb28m8;  C(1,3,64) = Cb38m8 ;
  C(0,0,65) = Ca08m7 ; C(0,1,65) = Ca18m7; C(0,2,65) =  Ca28m7;  C(0,3,65) = Ca38m7 ;
  C(1,0,65) = Cb08m7 ; C(1,1,65) = Cb18m7; C(1,2,65) =  Cb28m7;  C(1,3,65) = Cb38m7 ;
  C(0,0,66) = Ca08m6 ; C(0,1,66) = Ca18m6; C(0,2,66) =  Ca28m6;  C(0,3,66) = Ca38m6 ;
  C(1,0,66) = Cb08m6 ; C(1,1,66) = Cb18m6; C(1,2,66) =  Cb28m6;  C(1,3,66) = Cb38m6 ;
  C(0,0,67) = Ca08m5 ; C(0,1,67) = Ca18m5; C(0,2,67) =  Ca28m5;  C(0,3,67) = Ca38m5 ;
  C(1,0,67) = Cb08m5 ; C(1,1,67) = Cb18m5; C(1,2,67) =  Cb28m5;  C(1,3,67) = Cb38m5 ;
  C(0,0,68) = Ca08m4 ; C(0,1,68) = Ca18m4; C(0,2,68) =  Ca28m4;  C(0,3,68) = Ca38m4 ;
  C(1,0,68) = Cb08m4 ; C(1,1,68) = Cb18m4; C(1,2,68) =  Cb28m4;  C(1,3,68) = Cb38m4 ;
  C(0,0,69) = Ca08m3 ; C(0,1,69) = Ca18m3; C(0,2,69) =  Ca28m3;  C(0,3,69) = Ca38m3 ;
  C(1,0,69) = Cb08m3 ; C(1,1,69) = Cb18m3; C(1,2,69) =  Cb28m3;  C(1,3,69) = Cb38m3 ;
  C(0,0,70) = Ca08m2 ; C(0,1,70) = Ca18m2; C(0,2,70) =  Ca28m2;  C(0,3,70) = Ca38m2 ;
  C(1,0,70) = Cb08m2 ; C(1,1,70) = Cb18m2; C(1,2,70) =  Cb28m2;  C(1,3,70) = Cb38m2 ;
  C(0,0,71) = Ca08m1 ; C(0,1,71) = Ca18m1; C(0,2,71) =  Ca28m1;  C(0,3,71) = Ca38m1 ;
  C(1,0,71) = Cb08m1 ; C(1,1,71) = Cb18m1; C(1,2,71) =  Cb28m1;  C(1,3,71) = Cb38m1 ;
  C(0,0,72) = Ca080  ; C(0,1,72) = Ca180 ; C(0,2,72) =  Ca280 ;  C(0,3,72) = Ca380  ;
  C(1,0,72) = Cb080  ; C(1,1,72) = Cb180 ; C(1,2,72) =  Cb280 ;  C(1,3,72) = Cb380  ;
  C(0,0,73) = Ca08p1 ; C(0,1,73) = Ca18p1; C(0,2,73) =  Ca28p1;  C(0,3,73) = Ca38p1 ;
  C(1,0,73) = Cb08p1 ; C(1,1,73) = Cb18p1; C(1,2,73) =  Cb28p1;  C(1,3,73) = Cb38p1 ;
  C(0,0,74) = Ca08p2 ; C(0,1,74) = Ca18p2; C(0,2,74) =  Ca28p2;  C(0,3,74) = Ca38p2 ;
  C(1,0,74) = Cb08p2 ; C(1,1,74) = Cb18p2; C(1,2,74) =  Cb28p2;  C(1,3,74) = Cb38p2 ;
  C(0,0,75) = Ca08p3 ; C(0,1,75) = Ca18p3; C(0,2,75) =  Ca28p3;  C(0,3,75) = Ca38p3 ;
  C(1,0,75) = Cb08p3 ; C(1,1,75) = Cb18p3; C(1,2,75) =  Cb28p3;  C(1,3,75) = Cb38p3 ;
  C(0,0,76) = Ca08p4 ; C(0,1,76) = Ca18p4; C(0,2,76) =  Ca28p4;  C(0,3,76) = Ca38p4 ;
  C(1,0,76) = Cb08p4 ; C(1,1,76) = Cb18p4; C(1,2,76) =  Cb28p4;  C(1,3,76) = Cb38p4 ;
  C(0,0,77) = Ca08p5 ; C(0,1,77) = Ca18p5; C(0,2,77) =  Ca28p5;  C(0,3,77) = Ca38p5 ;
  C(1,0,77) = Cb08p5 ; C(1,1,77) = Cb18p5; C(1,2,77) =  Cb28p5;  C(1,3,77) = Cb38p5 ;
  C(0,0,78) = Ca08p6 ; C(0,1,78) = Ca18p6; C(0,2,78) =  Ca28p6;  C(0,3,78) = Ca38p6 ;
  C(1,0,78) = Cb08p6 ; C(1,1,78) = Cb18p6; C(1,2,78) =  Cb28p6;  C(1,3,78) = Cb38p6 ;
  C(0,0,79) = Ca08p7 ; C(0,1,79) = Ca18p7; C(0,2,79) =  Ca28p7;  C(0,3,79) = Ca38p7 ;
  C(1,0,79) = Cb08p7 ; C(1,1,79) = Cb18p7; C(1,2,79) =  Cb28p7;  C(1,3,79) = Cb38p7 ;
  C(0,0,80) = Ca08p8 ; C(0,1,80) = Ca18p8; C(0,2,80) =  Ca28p8;  C(0,3,80) = Ca38p8 ;
  C(1,0,80) = Cb08p8 ; C(1,1,80) = Cb18p8; C(1,2,80) =  Cb28p8;  C(1,3,80) = Cb38p8 ;
  C(0,0,81) = Ca09m9 ; C(0,1,81) = Ca19m9; C(0,2,81) =  Ca29m9;  C(0,3,81) = Ca39m9 ;
  C(1,0,81) = Cb09m9 ; C(1,1,81) = Cb19m9; C(1,2,81) =  Cb29m9;  C(1,3,81) = Cb39m9 ;
  C(0,0,82) = Ca09m8 ; C(0,1,82) = Ca19m8; C(0,2,82) =  Ca29m8;  C(0,3,82) = Ca39m8 ;
  C(1,0,82) = Cb09m8 ; C(1,1,82) = Cb19m8; C(1,2,82) =  Cb29m8;  C(1,3,82) = Cb39m8 ;
  C(0,0,83) = Ca09m7 ; C(0,1,83) = Ca19m7; C(0,2,83) =  Ca29m7;  C(0,3,83) = Ca39m7 ;
  C(1,0,83) = Cb09m7 ; C(1,1,83) = Cb19m7; C(1,2,83) =  Cb29m7;  C(1,3,83) = Cb39m7 ;
  C(0,0,84) = Ca09m6 ; C(0,1,84) = Ca19m6; C(0,2,84) =  Ca29m6;  C(0,3,84) = Ca39m6 ;
  C(1,0,84) = Cb09m6 ; C(1,1,84) = Cb19m6; C(1,2,84) =  Cb29m6;  C(1,3,84) = Cb39m6 ;
  C(0,0,85) = Ca09m5 ; C(0,1,85) = Ca19m5; C(0,2,85) =  Ca29m5;  C(0,3,85) = Ca39m5 ;
  C(1,0,85) = Cb09m5 ; C(1,1,85) = Cb19m5; C(1,2,85) =  Cb29m5;  C(1,3,85) = Cb39m5 ;
  C(0,0,86) = Ca09m4 ; C(0,1,86) = Ca19m4; C(0,2,86) =  Ca29m4;  C(0,3,86) = Ca39m4 ;
  C(1,0,86) = Cb09m4 ; C(1,1,86) = Cb19m4; C(1,2,86) =  Cb29m4;  C(1,3,86) = Cb39m4 ;
  C(0,0,87) = Ca09m3 ; C(0,1,87) = Ca19m3; C(0,2,87) =  Ca29m3;  C(0,3,87) = Ca39m3 ;
  C(1,0,87) = Cb09m3 ; C(1,1,87) = Cb19m3; C(1,2,87) =  Cb29m3;  C(1,3,87) = Cb39m3 ;
  C(0,0,88) = Ca09m2 ; C(0,1,88) = Ca19m2; C(0,2,88) =  Ca29m2;  C(0,3,88) = Ca39m2 ;
  C(1,0,88) = Cb09m2 ; C(1,1,88) = Cb19m2; C(1,2,88) =  Cb29m2;  C(1,3,88) = Cb39m2 ;
  C(0,0,89) = Ca09m1 ; C(0,1,89) = Ca19m1; C(0,2,89) =  Ca29m1;  C(0,3,89) = Ca39m1 ;
  C(1,0,89) = Cb09m1 ; C(1,1,89) = Cb19m1; C(1,2,89) =  Cb29m1;  C(1,3,89) = Cb39m1 ;
  C(0,0,90) = Ca090  ; C(0,1,90) = Ca190 ; C(0,2,90) =  Ca290 ;  C(0,3,90) = Ca390  ;
  C(1,0,90) = Cb090  ; C(1,1,90) = Cb190 ; C(1,2,90) =  Cb290 ;  C(1,3,90) = Cb390  ;
  C(0,0,91) = Ca09p1 ; C(0,1,91) = Ca19p1; C(0,2,91) =  Ca29p1;  C(0,3,91) = Ca39p1 ;
  C(1,0,91) = Cb09p1 ; C(1,1,91) = Cb19p1; C(1,2,91) =  Cb29p1;  C(1,3,91) = Cb39p1 ;
  C(0,0,92) = Ca09p2 ; C(0,1,92) = Ca19p2; C(0,2,92) =  Ca29p2;  C(0,3,92) = Ca39p2 ;
  C(1,0,92) = Cb09p2 ; C(1,1,92) = Cb19p2; C(1,2,92) =  Cb29p2;  C(1,3,92) = Cb39p2 ;
  C(0,0,93) = Ca09p3 ; C(0,1,93) = Ca19p3; C(0,2,93) =  Ca29p3;  C(0,3,93) = Ca39p3 ;
  C(1,0,93) = Cb09p3 ; C(1,1,93) = Cb19p3; C(1,2,93) =  Cb29p3;  C(1,3,93) = Cb39p3 ;
  C(0,0,94) = Ca09p4 ; C(0,1,94) = Ca19p4; C(0,2,94) =  Ca29p4;  C(0,3,94) = Ca39p4 ;
  C(1,0,94) = Cb09p4 ; C(1,1,94) = Cb19p4; C(1,2,94) =  Cb29p4;  C(1,3,94) = Cb39p4 ;
  C(0,0,95) = Ca09p5 ; C(0,1,95) = Ca19p5; C(0,2,95) =  Ca29p5;  C(0,3,95) = Ca39p5 ;
  C(1,0,95) = Cb09p5 ; C(1,1,95) = Cb19p5; C(1,2,95) =  Cb29p5;  C(1,3,95) = Cb39p5 ;
  C(0,0,96) = Ca09p6 ; C(0,1,96) = Ca19p6; C(0,2,96) =  Ca29p6;  C(0,3,96) = Ca39p6 ;
  C(1,0,96) = Cb09p6 ; C(1,1,96) = Cb19p6; C(1,2,96) =  Cb29p6;  C(1,3,96) = Cb39p6 ;
  C(0,0,97) = Ca09p7 ; C(0,1,97) = Ca19p7; C(0,2,97) =  Ca29p7;  C(0,3,97) = Ca39p7 ;
  C(1,0,97) = Cb09p7 ; C(1,1,97) = Cb19p7; C(1,2,97) =  Cb29p7;  C(1,3,97) = Cb39p7 ;
  C(0,0,98) = Ca09p8 ; C(0,1,98) = Ca19p8; C(0,2,98) =  Ca29p8;  C(0,3,98) = Ca39p8 ;
  C(1,0,98) = Cb09p8 ; C(1,1,98) = Cb19p8; C(1,2,98) =  Cb29p8;  C(1,3,98) = Cb39p8 ;
  C(0,0,99) = Ca09p9 ; C(0,1,99) = Ca19p9; C(0,2,99) =  Ca29p9;  C(0,3,99) = Ca39p9 ;
  C(1,0,99) = Cb09p9 ; C(1,1,99) = Cb19p9; C(1,2,99) =  Cb29p9;  C(1,3,99) = Cb39p9 ;


//C.t().print(" ");

//  cout << "Part 4: Done" << endl;
  //----------------------------------------------------------------------------------------------------------------
  // Part 5) get Power Spectrum-> Gauss quaduature used in 3D. Int Tnml(r,The,Phi) rho(r,theta,phi) dV
  //----------------------------------------------------------------------------------------------------------------

  
  
  

 int incrementN = 0; 

// cube P = zeros<cube>(3,3,10);
  double P[2][3][3][10]; // Power Spectrum P[A-type][n1][n2][l]
  memset(P, 0.0, sizeof P);

  for(int a=0; a < 2; a++){ 
    for(int n1=0; n1 < 3; n1++){ 
      for(int n2=0; n2 < 3; n2++){ 

          incrementN = 0;

        for(int l=0; l <= 9; l++){ 
          for(int m=-l; m <= l; m++){ 
            P[a][n1][n2][l] += C(a,n1,incrementN)*C(a,n2,incrementN);
            incrementN++;
          }
            cout << P[a][n1][n2][l] << endl;
        }
      }
    }
  }

//  cout << size(C) << endl;
//  cout << "Part 5: Done" << endl;

  
//P.print("P");


return 0;
}



































