#include <Eigen/Core>
#include <Spectra/GenEigsSolver.h>
// <Spectra/MatOp/DenseGenMatProd.h> is implicitly included
#include <iostream>
using namespace Spectra;
int main()
{
    // We are going to calculate the eigenvalues of M
    Eigen::MatrixXd M = Eigen::MatrixXd::Random(10, 10);
    // Construct matrix operation object using the wrapper class
    DenseGenMatProd<double> op(M);
    // Construct eigen solver object, requesting the largest
    // (in magnitude, or norm) three eigenvalues
    GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, 3, 6);
    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();
    // Retrieve results
    Eigen::VectorXcd evalues;
    if(eigs.info() == SUCCESSFUL)
        evalues = eigs.eigenvalues();
    std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    return 0;
}