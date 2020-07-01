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

#include "veins/modules/application/platooning/detection/BeaconAnalyzer.h"

BeaconAnalyzer::BeaconAnalyzer(const PlatooningParameters& platooningParameters, const DetectionParameters& detectionParameters,
                               const std::map<Plexe::VEHICLE_SENSORS, SensorParameters*>& sensorParameters, double qFactor)
    : platooningParameters(platooningParameters)
    , kfPredecessorTime(0)
    , kfFollowerTime(0)
{

    // Obtain the errors associated with the sensor parameters
    double positionError = sensorParameters.at(Plexe::VEHICLE_SENSORS::EGO_GPS_X)->getAbsoluteError();
    double speedError = sensorParameters.at(Plexe::VEHICLE_SENSORS::EGO_SPEED)->getAbsoluteError();
    double radarDistanceError = sensorParameters.at(Plexe::VEHICLE_SENSORS::RADAR_DISTANCE)->getAbsoluteError();
    double radarSpeedError = sensorParameters.at(Plexe::VEHICLE_SENSORS::RADAR_SPEED)->getAbsoluteError();

    // The matrices used to initialize the Kalman Filters (A and B are just placehoders)
    const auto A = (Eigen::Matrix2d() << 1, 1, 1, 1).finished();
    const auto B = (Eigen::Vector2d() << 1, 1).finished();
    const auto H = Eigen::Matrix2d::Identity();

    P0 = Eigen::Vector2d(positionError * positionError, speedError * speedError).asDiagonal();
    const auto Q = P0 / (qFactor * qFactor);
    const auto R = P0;

    // Initialize the Kalman Filter objects
    kfPredecessor = std::make_shared<KalmanFilter>(A, B, H, Q, R);
    kfFollower = std::make_shared<KalmanFilter>(A, B, H, Q, R);

    // Initialize the objects used to compute the running averages
    distanceV2XKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    distanceRadarKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedV2XKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedRadarV2XAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedRadarKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);

    // Initialize the AttackDetection objects
    initializeDistanceDetectors(detectionParameters, radarDistanceError);
    initializeSpeedDetectors(detectionParameters, speedError, radarSpeedError);
}

bool BeaconAnalyzer::attackDetected() const
{
    return attackDetectedType() > 0;
}

int BeaconAnalyzer::attackDetectedType() const
{
    return
        (distanceKFDetector->attackDetected() ? 1 : 0) +
        (distanceV2XKFDetector->attackDetected() ? 2 : 0) +
        (speedV2XKFDetector->attackDetected() ? 4 : 0) +
        (distanceRadarDetector->attackDetected() ? 8 : 0) +
        (distanceRadarKFDetector->attackDetected() ? 16 : 0) +
        (speedRadarV2XDetector->attackDetected() ? 32 : 0) +
        (speedRadarKFDetector->attackDetected() ? 64 : 0);
}

void BeaconAnalyzer::update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData)
{
    if (cachedFollowerData) {
        update(predData, cachedFollowerData);
    }
}

void BeaconAnalyzer::update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData,
                            const std::shared_ptr<Plexe::VEHICLE_DATA>& follData,
                            const std::shared_ptr<Plexe::RADAR_READING>& radarReading)
{
    // Update the position of the predecessor depending on the time elapsed between the reception of the two beacons
    predData->positionX += predData->speed * (follData->time - predData->time);

    auto predEstimation = updateKalmanFilter(predData, kfPredecessor, kfPredecessorTime);
    auto follEstimation = updateKalmanFilter(follData, kfFollower, kfFollowerTime);

    double v2xDistance = predData->positionX - follData->positionX;
    double kfDistance = predEstimation.first(0) - follEstimation.first(0);
    double expectedDistance = computeExpectedDistance(follData->speed);

    // KF Distance
    distanceKFDetector->update(kfDistance - expectedDistance - predData->length, expectedDistance);

    // V2X Distance - KF Distance
    distanceV2XKFAvg->addValue(v2xDistance - kfDistance);
    distanceV2XKFDetector->update(distanceV2XKFAvg->getRunningAverage(),
                                  predEstimation.second(0, 0), follEstimation.second(0, 0));

    // V2X Speed - KF Speed
    speedV2XKFAvg->addValue(predData->speed - predEstimation.first(1));
    speedV2XKFDetector->update(speedV2XKFAvg->getRunningAverage(), predEstimation.second(1, 1), follData->acceleration);

    if (radarReading && radarReading->valid()) {
        // Radar Distance
        distanceRadarDetector->update(radarReading->distance - expectedDistance, expectedDistance);
        
        // Radar Distance - KF Distance
        distanceRadarKFAvg->addValue(radarReading->distance - kfDistance + predData->length);
        distanceRadarKFDetector->update(distanceRadarKFAvg->getRunningAverage(),
                                        predEstimation.second(0, 0), follEstimation.second(0, 0));

        double relativeSpeedV2X = predData->speed - follData->speed;
        double relativeSpeedKF = predEstimation.first(1) - follEstimation.first(1);

        // Radar Speed - V2X Speed
        speedRadarV2XAvg->addValue(radarReading->relativeSpeed - relativeSpeedV2X);
        speedRadarV2XDetector->update(speedRadarV2XAvg->getRunningAverage(), follData->acceleration);

        // Radar Speed - KF Speed
        speedRadarKFAvg->addValue(radarReading->relativeSpeed - relativeSpeedKF);
        speedRadarKFDetector->update(speedRadarKFAvg->getRunningAverage(), predEstimation.second(1, 1),
                                     follEstimation.second(1, 1), follData->acceleration);
    }
}

void BeaconAnalyzer::cacheFollowerData(const std::shared_ptr<Plexe::VEHICLE_DATA>& follData)
{
    cachedFollowerData = follData;
}

void BeaconAnalyzer::initializeDistanceDetectors(const DetectionParameters& detectionParameters, double radarDistanceError)
{
    //------------right elements of inequalities (distance)
    auto attackTolerance = detectionParameters.attackTolerance;

    auto distanceKFDetectorTh = [detectionParameters](double expectedDistance){
        return detectionParameters.distanceKFThresholdFactor * expectedDistance;
    };
    distanceKFDetector = std::make_shared<DistanceDetectorType1>(distanceKFDetectorTh, attackTolerance);

    auto distanceRadarDetectorTh = [detectionParameters](double expectedDistance){
        return detectionParameters.distanceRadarThresholdFactor * expectedDistance;
    };
    distanceRadarDetector = std::make_shared<DistanceDetectorType1>(distanceRadarDetectorTh, attackTolerance);

    auto distanceV2XKFDetectorTh = [detectionParameters](double kfVariancePred, double kfVarianceFoll){
        return 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll)) *
               detectionParameters.distanceV2XKFThresholdFactor;
    };
    distanceV2XKFDetector = std::make_shared<DistanceDetectorType2>(distanceV2XKFDetectorTh, attackTolerance);

    auto distanceRadarKFDetectorTh = [radarDistanceError, detectionParameters](double kfVariancePred, double kfVarianceFoll){
        return (radarDistanceError + 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll))) *
               detectionParameters.distanceRadarKFThresholdFactor;
    };
    distanceRadarKFDetector = std::make_shared<DistanceDetectorType2>(distanceRadarKFDetectorTh, attackTolerance);
}

void BeaconAnalyzer::initializeSpeedDetectors(const DetectionParameters& detectionParameters, double speedError, double radarSpeedError)
{
    //------------right elements of inequalities (speed)
    auto attackTolerance = detectionParameters.attackTolerance;

    auto speedV2XKFDetectorTh = [speedError, detectionParameters](double kfVariance, double acceleration){
        return detectionParameters.speedV2XKFThresholdFactor * (speedError + std::sqrt(kfVariance) * 3) *
               (1 + std::abs(acceleration) * detectionParameters.accelerationFactor);
    };
    speedV2XKFDetector = std::make_shared<SpeedDetectorType1>(speedV2XKFDetectorTh, attackTolerance);

    auto speedRadarV2XDetectorTh = [speedError, radarSpeedError, detectionParameters](double acceleration){
        return detectionParameters.speedRadarV2XThresholdFactor * (radarSpeedError + 2*speedError) *
               (1 + std::abs(acceleration) * detectionParameters.accelerationFactor);
    };
    speedRadarV2XDetector = std::make_shared<SpeedDetectorType2>(speedRadarV2XDetectorTh, attackTolerance);

    auto speedRadarKFDetectorTh = [radarSpeedError, detectionParameters](
            double kfVariancePred, double kfVarianceFoll, double acceleration){
        return detectionParameters.speedRadarKFThresholdFactor *
               (radarSpeedError + 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll))) *
               (1 + std::abs(acceleration) * detectionParameters.accelerationFactor);
    };
    speedRadarKFDetector = std::make_shared<SpeedDetectorType3>(speedRadarKFDetectorTh, attackTolerance);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> BeaconAnalyzer::updateKalmanFilter(
        const std::shared_ptr<Plexe::VEHICLE_DATA>& vehicleData, std::shared_ptr<KalmanFilter>& kalmanFilter, double& previousTime)
{
    double position = vehicleData->positionX;
    double speed = vehicleData->speed;
    double acceleration = vehicleData->acceleration;

    double dt = vehicleData->time - previousTime;
    previousTime = vehicleData->time;

    const auto A = (Eigen::Matrix2d() << 1, dt, 0, 1).finished();
    const auto B = (Eigen::Vector2d() << 0.5 * dt * dt, dt).finished();

    const auto u = (Eigen::VectorXd(1) << acceleration).finished();
    const auto ym = (Eigen::Vector2d() << position, speed).finished();

    if (kalmanFilter->isInitialized()) {
        kalmanFilter->predict(u, A, B);
        kalmanFilter->update(ym);
    }
    else {
        kalmanFilter->initialize(ym, P0);
    }

    return kalmanFilter->getEstimation();
}

double BeaconAnalyzer::computeExpectedDistance(double speed) const
{
    return platooningParameters.spacing + platooningParameters.headway * speed;
}