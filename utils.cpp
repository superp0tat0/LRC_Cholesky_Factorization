#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

namespace MatrixOps{
    MatrixXcd Reduce_Cholesky(MatrixXd A){
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
        return(r);
    }
}