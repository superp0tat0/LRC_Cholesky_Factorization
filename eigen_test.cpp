#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <vector>
#include <chrono>
#include <ctime>

using namespace std;
using namespace Eigen;
using namespace Spectra;


MatrixXcd Reduce_Cholesky(MatrixXd A){
    // The machine precision, ~= 1e-16 for the "double" type
    //Get the Eigen value and the Eigen vectors of the Matrix
    EigenSolver<MatrixXd> es(A);
    VectorXd evl = es.eigenvalues().real();
    MatrixXd evc = es.eigenvectors().real();

    //Use the first loop to get the correct size of the entry
    int size = 0;
    for (Eigen::Index i=0; i<evl.size(); ++i) {
        if(evl(i,0) >= 1e-15){
            ++size;
        }
    }

    //Use the second loop the get the correct non-zero indexs
    std::vector<Eigen::Index> idxs(size);
    int counter = 0;

    for (Eigen::Index i=0; i<evl.size(); ++i) {
        if(evl(i,0) >= 1e-15){
            idxs[counter] = i;
            ++counter;
        }
    }

    //Get the selected eigenvectors and the eigenvalues, perform QR decomposition
    MatrixXd evl_matrix = evl(idxs).asDiagonal().toDenseMatrix().cwiseSqrt();
    MatrixXd S = ( evl_matrix * evc(idxs, Eigen::all) ).transpose();

    HouseholderQR<MatrixXcd> qr(S.rows(), S.cols());
    qr.compute(S);
    // MatrixXd q = qr.householderQ()* MatrixXd::Identity(S.rows(), S.cols());
    MatrixXcd temp = qr.matrixQR().triangularView<Upper>();
    MatrixXcd r = temp.topRows(S.cols());
    return r;
}

int main(){
    MatrixXd C = MatrixXd::Random(500,100);
    MatrixXd A = C * C.transpose();

    auto start = std::chrono::system_clock::now();

    FullPivLU<MatrixXd> lu_decomp(A);
    //cout << "The rank of A is " << lu_decomp.rank() << endl;

    // Construct matrix operation object using the wrapper class
    DenseGenMatProd<double> op(A);
    GenEigsSolver<DenseGenMatProd<double>> eigs(op, lu_decomp.rank(), 2*lu_decomp.rank());

    eigs.init();
    int nconv = eigs.compute();
    
    Eigen::VectorXd evalues(lu_decomp.rank());
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues().real();
    //std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    
    return 0;
}