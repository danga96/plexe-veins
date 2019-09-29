//
// Copyright (C) 2019 Marco Iorio <marco.iorio@polito.it>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#include <stdexcept>

#include "KalmanFilter.h"

KalmanFilter::KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
    : A(A)
    , B(B)
    , H(H)
    , I(Eigen::MatrixXd::Identity(A.rows(), A.rows()))
    , Q(Q)
    , R(R)
    , initialized(false)
{
}

void KalmanFilter::initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& p0)
{
    X = x0;
    P = p0;
    initialized = true;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::predict(const Eigen::VectorXd& u)
{
    if (!isInitialized()) {
        throw std::runtime_error("The Kalman filter has not been initialized");
    }

    // Compute the prediction for the state X
    Xp = A * X + B * u;

    // Update the State covariance matrix P
    P = A * P * A.transpose() + Q;

    return {Xp, P};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::predict(const Eigen::VectorXd& u, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
    this->A = A;
    this->B = B;
    return predict(u);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::update(const Eigen::VectorXd& ym)
{
    if (!isInitialized()) {
        throw std::runtime_error("The Kalman filter has not been initialized");
    }

    // Compute the Kalman gains
    K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

    // Update the estimation
    X = Xp + K * (ym - H * Xp);

    // Update the process covariance matrix
    P = (I - K * H) * P;

    return {X, P};
}