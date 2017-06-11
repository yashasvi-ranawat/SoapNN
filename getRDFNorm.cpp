#include<iostream>
#include<armadillo>
#include"myArmadillo.h"
#include"myMath.h"
#include"mySoap.h"

using namespace std;
using namespace arma;


int main(int argc, char** argv) {

  
  double pi = 3.14159265358979324;
  double halfPi = 3.14159265358979324*0.5;


  double rcut = 100.0;
  double ao = 2.0;
  double z = 1.0;
  double norm = pow(sqrt(z/ao),3);
  double rsc = 0.5*rcut; // rescaleing the integration for gauss-legendre quaduature.

  double a0 = 0.5;
  
  mat GL; // [http://keisan.casio.com/exec/system/1329114617 (June 5th 2017)] , produced by Octave. W(:,0) -> GL coord. pos. W(:,1) -> GL weights.
  GL.load("parameters100.txt");
//  GL.load("P500_both.txt");

  vec R = rcut*0.5*GL.col(0) + rcut*0.5 ; // rescaled R for gauss-legendre quaduature


vec rdf1 = hydrogenRDF(1, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
vec rdf2 = hydrogenRDF(2, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
vec rdf3 = hydrogenRDF(3, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)
vec rdf4 = hydrogenRDF(4, z,ao,norm,R); // a0 = 0.5, Norm. Const. = 2^(3/2)

R.save("R.dat",raw_ascii);
rdf1.save("rdf1.dat",raw_ascii);
rdf2.save("rdf2.dat",raw_ascii);
rdf3.save("rdf3.dat",raw_ascii);
rdf4.save("rdf4.dat",raw_ascii);

cout << integ1D(R,R%R%rdf1%rdf1,GL, rcut) << endl; // int r^2 rdf1^2 dr
cout << integ1D(R,R%R%rdf2%rdf2,GL, rcut) << endl; // int r^2 rdf2^2 dr
cout << integ1D(R,R%R%rdf3%rdf3,GL, rcut) << endl; // ...
cout << integ1D(R,R%R%rdf4%rdf4,GL, rcut) << endl; // ...

cout << integ1D(R,R%R%rdf1%rdf2,GL, rcut) << endl; // int r^2 rdf1*rdf2 dr
cout << integ1D(R,R%R%rdf1%rdf3,GL, rcut) << endl; // int r^2 rdf1*rdf3 dr
cout << integ1D(R,R%R%rdf2%rdf3,GL, rcut) << endl; // int r^2 rdf2*rdf3 dr
cout << integ1D(R,R%R%rdf1%rdf4,GL, rcut) << endl; // ...
cout << integ1D(R,R%R%rdf2%rdf4,GL, rcut) << endl; // ...
cout << integ1D(R,R%R%rdf3%rdf4,GL, rcut) << endl; // ...

rdf4.print("rdf4");


return 0;
}
