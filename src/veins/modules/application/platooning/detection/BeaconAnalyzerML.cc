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
#include <fdeep/fdeep.hpp>
#include "veins/modules/application/platooning/detection/BeaconAnalyzerML.h"

BeaconAnalyzerML::BeaconAnalyzerML(const PlatooningParameters& platooningParameters, const DetectionParameters& detectionParameters, const std::map<Plexe::VEHICLE_SENSORS, SensorParameters*>& sensorParameters, double qFactor)
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
    distanceKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    distanceRadarAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    distanceV2XKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    distanceRadarKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);

    speedV2XKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedRadarV2XAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedRadarKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);
    speedKFAvg = std::make_shared<RunningAverage<double>>(detectionParameters.runningAvgWindow);

    // Initialize the AttackDetection objects
    initializeDistanceDetectors(detectionParameters, radarDistanceError);
    initializeSpeedDetectors(detectionParameters, speedError, radarSpeedError);

    // Initialize and load model
    //const auto modell = fdeep::load_model("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_V2XKFdistance.json", true, fdeep::dev_null_logger);
}

bool BeaconAnalyzerML::attackDetected() const
{
    return attackDetectedType() > 0;
}

int BeaconAnalyzerML::attackDetectedType() const
{
    return (distanceKFDetectorML->attackDetected() ? 1 : 0) + 
            (distanceV2XKFDetectorML->attackDetected() ? 2 : 0) +
            (speedV2XKFDetectorML->attackDetected() ? 4 : 0) + 
            (distanceRadarDetectorML->attackDetected() ? 8 : 0) + 
            (distanceRadarKFDetectorML->attackDetected() ? 16 : 0) + 
            (speedRadarV2XDetectorML->attackDetected() ? 32 : 0) + 
            (speedRadarKFDetectorML->attackDetected() ? 64 : 0) +
            (speedKFDetectorML->attackDetected() ? 128 : 0);
}

void BeaconAnalyzerML::update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData)
{
    if (cachedFollowerData) {
        update(predData, cachedFollowerData);
    }
}

void BeaconAnalyzerML::test_ML(std::vector<double> value)
{
    std::vector<float> floatVec(value.begin(), value.end());
    std::cout << "Time " << simTime().dbl() << " Value: " << value[0] << " SIZE: " << value.size() << endl;
    const auto model = fdeep::load_model("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_V2XKFdistance.json");

    const fdeep::tensor t(fdeep::tensor_shape(10, 1), floatVec);
    const auto result = model.predict({t});
    std::cout << "TEST2" << fdeep::show_tensors(result) << std::endl;
}

void BeaconAnalyzerML::update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData, const std::shared_ptr<Plexe::VEHICLE_DATA>& follData, const std::shared_ptr<Plexe::RADAR_READING>& radarReading)
{

    // Update the position of the predecessor depending on the time elapsed between the reception of the two beacons
    predData->positionX += predData->speed * (follData->time - predData->time);

    auto predEstimation = updateKalmanFilter(predData, kfPredecessor, kfPredecessorTime);
    auto follEstimation = updateKalmanFilter(follData, kfFollower, kfFollowerTime);

    double v2xDistance = predData->positionX - follData->positionX;
    double kfDistance = predEstimation.first(0) - follEstimation.first(0);
    double expectedDistance = computeExpectedDistance(follData->speed);
    
    // KF Distance
    //distanceKFDetector->update(kfDistance - expectedDistance - predData->length, expectedDistance);
    distanceKFAvg->addValue(kfDistance - expectedDistance - predData->length);
    distanceKFDetectorML->update(distanceKFAvg->getBuffer(),simTime().dbl());
    
    // V2X Distance - KF Distance
    distanceV2XKFAvg->addValue(v2xDistance - kfDistance);
    /*bool var = */ 
    //distanceV2XKFDetector->update(distanceV2XKFAvg->getRunningAverage(), predEstimation.second(0, 0), follEstimation.second(0, 0));
    // std::cout<<"Time "<<simTime().dbl()<<" pred: "<<predEstimation.first(0)<<" foll: "<<follEstimation.first(0)<<" var:"<<var<<endl;
    //test_ML(distanceV2XKFAvg->getBuffer());
    distanceV2XKFDetectorML->update(distanceV2XKFAvg->getRunningAverage_mod(),simTime().dbl());
    
    // V2X Speed - KF Speed
    speedV2XKFAvg->addValue(predData->speed - predEstimation.first(1));
    //speedV2XKFDetector->update(speedV2XKFAvg->getRunningAverage(), predEstimation.second(1, 1), follData->acceleration);
    speedV2XKFDetectorML->update(speedV2XKFAvg->getRunningAverage_mod(),simTime().dbl());
    
    // KF - Speed
    speedKFAvg->addValue(predEstimation.first(1) - follEstimation.first(1));
    speedKFDetectorML->update(speedKFAvg->getRunningAverage_mod(),simTime().dbl());
    

    if (radarReading && radarReading->valid()) {
        // Radar Distance
        distanceRadarAvg->addValue(radarReading->distance - expectedDistance);
        //distanceRadarDetector->update(radarReading->distance - expectedDistance, expectedDistance);
        distanceRadarDetectorML->update(distanceRadarAvg->getBuffer(),simTime().dbl());
        
        // Radar Distance - KF Distance
        distanceRadarKFAvg->addValue(radarReading->distance - kfDistance + predData->length);
        //distanceRadarKFDetector->update(distanceRadarKFAvg->getRunningAverage(), predEstimation.second(0, 0), follEstimation.second(0, 0));
        distanceRadarKFDetectorML->update(distanceRadarKFAvg->getRunningAverage_mod(),simTime().dbl());
        
        double relativeSpeedV2X = predData->speed - follData->speed;
        double relativeSpeedKF = predEstimation.first(1) - follEstimation.first(1);

        // Radar Speed - V2X Speed
        speedRadarV2XAvg->addValue(radarReading->relativeSpeed - relativeSpeedV2X);
        //speedRadarV2XDetector->update(speedRadarV2XAvg->getRunningAverage(), follData->acceleration);
        speedRadarV2XDetectorML->update(speedRadarV2XAvg->getRunningAverage_mod(),simTime().dbl());
        
        // Radar Speed - KF Speed
        speedRadarKFAvg->addValue(radarReading->relativeSpeed - relativeSpeedKF);
        //speedRadarKFDetector->update(speedRadarKFAvg->getRunningAverage(), predEstimation.second(1, 1), follEstimation.second(1, 1), follData->acceleration);
        speedRadarKFDetectorML->update(speedRadarKFAvg->getRunningAverage_mod(),simTime().dbl());
        
    }
}

void BeaconAnalyzerML::cacheFollowerData(const std::shared_ptr<Plexe::VEHICLE_DATA>& follData)
{
    cachedFollowerData = follData;
}

void BeaconAnalyzerML::initializeDistanceDetectors(const DetectionParameters& detectionParameters, double radarDistanceError)
{
    //------------right elements of inequalities (distance)
    //auto attackTolerance = detectionParameters.attackTolerance;
    double ML_accuracy = detectionParameters.ML_accuracy;
    std::string ML_model_path = detectionParameters.ML_model_path;
    
    std::cout << "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" << endl;
    //auto distanceKFDetectorTh = [detectionParameters](double expectedDistance) { return detectionParameters.distanceKFThresholdFactor * expectedDistance; };
    //distanceKFDetector = std::make_shared<DistanceDetectorType1>(distanceKFDetectorTh, attackTolerance);
    distanceKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"KFdistance",ML_model_path);



    //auto distanceRadarDetectorTh = [detectionParameters](double expectedDistance) { return detectionParameters.distanceRadarThresholdFactor * expectedDistance; };
    //distanceRadarDetector = std::make_shared<DistanceDetectorType1>(distanceRadarDetectorTh, attackTolerance);
    distanceRadarDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"Rdistance",ML_model_path);

    //auto distanceV2XKFDetectorTh = [detectionParameters](double kfVariancePred, double kfVarianceFoll) { return 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll)) * detectionParameters.distanceV2XKFThresholdFactor; };
    //distanceV2XKFDetector = std::make_shared<DistanceDetectorType2>(distanceV2XKFDetectorTh, attackTolerance);
    //auto model = fdeep::load_model("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_V2XKFdistance.json", true, fdeep::dev_null_logger);
    distanceV2XKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"V2XKFdistance",ML_model_path);

    //auto distanceRadarKFDetectorTh = [radarDistanceError, detectionParameters](double kfVariancePred, double kfVarianceFoll) { return (radarDistanceError + 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll))) * detectionParameters.distanceRadarKFThresholdFactor; };
    //distanceRadarKFDetector = std::make_shared<DistanceDetectorType2>(distanceRadarKFDetectorTh, attackTolerance);
    distanceRadarKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"RKFdistance",ML_model_path);
}

void BeaconAnalyzerML::initializeSpeedDetectors(const DetectionParameters& detectionParameters, double speedError, double radarSpeedError)
{
    //------------right elements of inequalities (speed)
    //auto attackTolerance = detectionParameters.attackTolerance;
    double ML_accuracy = detectionParameters.ML_accuracy;
    std::string ML_model_path = detectionParameters.ML_model_path;

    //auto speedV2XKFDetectorTh = [speedError, detectionParameters](double kfVariance, double acceleration) { return detectionParameters.speedV2XKFThresholdFactor * (speedError + std::sqrt(kfVariance) * 3) * (1 + std::abs(acceleration) * detectionParameters.accelerationFactor); };
    //speedV2XKFDetector = std::make_shared<SpeedDetectorType1>(speedV2XKFDetectorTh, attackTolerance);
    speedV2XKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"V2XKFspeed",ML_model_path);

    //auto speedRadarV2XDetectorTh = [speedError, radarSpeedError, detectionParameters](double acceleration) { return detectionParameters.speedRadarV2XThresholdFactor * (radarSpeedError + 2 * speedError) * (1 + std::abs(acceleration) * detectionParameters.accelerationFactor); };
    //speedRadarV2XDetector = std::make_shared<SpeedDetectorType2>(speedRadarV2XDetectorTh, attackTolerance);
    speedRadarV2XDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"RV2Xspeed",ML_model_path);

    //auto speedRadarKFDetectorTh = [radarSpeedError, detectionParameters](double kfVariancePred, double kfVarianceFoll, double acceleration) { return detectionParameters.speedRadarKFThresholdFactor * (radarSpeedError + 3 * (std::sqrt(kfVariancePred) + std::sqrt(kfVarianceFoll))) * (1 + std::abs(acceleration) * detectionParameters.accelerationFactor); };
    //speedRadarKFDetector = std::make_shared<SpeedDetectorType3>(speedRadarKFDetectorTh, attackTolerance);
    speedRadarKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"RKFspeed",ML_model_path);

    speedKFDetectorML = std::make_shared<AttackDetectorML>(ML_accuracy,"KFspeed",ML_model_path);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> BeaconAnalyzerML::updateKalmanFilter(const std::shared_ptr<Plexe::VEHICLE_DATA>& vehicleData, std::shared_ptr<KalmanFilter>& kalmanFilter, double& previousTime)
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
 
double BeaconAnalyzerML::computeExpectedDistance(double speed) const
{
    return platooningParameters.spacing + platooningParameters.headway * speed;
}