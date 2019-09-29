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

#include <eigen3/Eigen/Dense>

#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

class KalmanFilter {

public:
    /**
     * Creates a new Kalman filter and initialized the specified matrices
     * @param A: system dynamics model
     * @param B: control-input model
     * @param H: observation model
     * @param Q: process noise covariance matrix
     * @param R: measurement covariance matrix
     */
    KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

    /**
     * Initializes the filter with a guess for the initial state and the associate covariance matrix
     * @param x0: the guess for the initial state
     * @param p0: the guess for the initial state covariance matrix
     */
    void initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& p0);

    /**
     * Predicts the new value of the state vector depending on the applied input
     * @param u: the input applied to the system
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::VectorXd& u);

    /**
     * Predicts the new value of the state vector depending on the applied input and the specified A and B matrices
     * @param u: the input applied to the system
     * @param A: system dynamics model
     * @param B: control-input model
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::VectorXd& u, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

    /**
     * Updates the estimated state based on the measured values
     * @param y: the measured values
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const Eigen::VectorXd& ym);

    /**
     * Returns the estimated state vestor and the associated covariance matrix
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> getEstimation() const
    {
        return {X, P};
    }

    /**
     * Returns whether the Kalman filter has already been initialized
     */
    bool isInitialized()
    {
        return initialized;
    }

private:
    Eigen::VectorXd X; // State vector
    Eigen::VectorXd Xp; // Predicted state vector
    Eigen::VectorXd Y; // Observation vector

    Eigen::MatrixXd P; // State covariance matrix
    Eigen::MatrixXd K; // Kalman gains

    Eigen::MatrixXd A; // System dynamics model (how the previous state evolves to the new one in absence of inputs)
    Eigen::MatrixXd B; // Control-input model (how the inputs influence the the evolution of the state)
    Eigen::MatrixXd H; // Observation model (maps the true state space into the observed space)
    Eigen::MatrixXd I; // The identity matrix

    Eigen::MatrixXd Q; // Process noise covariance matrix
    Eigen::MatrixXd R; // Measurement covariance matrix

    bool initialized; // Whether the Kalman Filter has already been initialized
};

#endif // KALMANFILTER_H_