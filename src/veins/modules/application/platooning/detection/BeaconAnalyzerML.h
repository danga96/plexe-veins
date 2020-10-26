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

#ifndef BEACONANALYZERML_H
#define BEACONANALYZERML_H

#include <memory>
#include <fdeep/fdeep.hpp>
#include "veins/modules/application/platooning/CC_Const.h"
#include "veins/modules/application/platooning/detection/AttackDetector.h"
#include "veins/modules/application/platooning/detection/AttackDetectorML.h"
#include "veins/modules/application/platooning/detection/KalmanFilter.h"
#include "veins/modules/application/platooning/detection/RunningAverage.h"
#include "veins/modules/application/platooning/sensors/SensorParameters.h"

class BeaconAnalyzerML {

public:
    struct PlatooningParameters {
        double spacing;
        double headway;
    };

    struct DetectionParameters {
        std::size_t runningAvgWindow;
        std::size_t attackTolerance;

        double ML_accuracy;

        double distanceKFThresholdFactor;
        double distanceRadarThresholdFactor;
        double distanceV2XKFThresholdFactor;
        double distanceRadarKFThresholdFactor;

        double speedV2XKFThresholdFactor;
        double speedRadarV2XThresholdFactor;
        double speedRadarKFThresholdFactor;

        double accelerationFactor;

        std::string ML_model_path;
    };

public:
    BeaconAnalyzerML(const PlatooningParameters& platooningParameters, const DetectionParameters& detectionParameters, const std::map<Plexe::VEHICLE_SENSORS, SensorParameters*>& sensorParameters, double qFactor);

    bool attackDetected() const;
    int attackDetectedType() const;

    void test_ML(std::vector<double> value);

    void update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData);

    void update(const std::shared_ptr<Plexe::VEHICLE_DATA>& predData, const std::shared_ptr<Plexe::VEHICLE_DATA>& follData, const std::shared_ptr<Plexe::RADAR_READING>& radarReading = nullptr);

    void cacheFollowerData(const std::shared_ptr<Plexe::VEHICLE_DATA>& follData);

private:
    void initializeDistanceDetectors(const DetectionParameters& detectionParameters, double radarDistanceError);
    void initializeSpeedDetectors(const DetectionParameters& detectionParameters, double speedError, double radarSpeedError);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> updateKalmanFilter(const std::shared_ptr<Plexe::VEHICLE_DATA>& vehicleData, std::shared_ptr<KalmanFilter>& kalmanFilter, double& previousTime);

    double computeExpectedDistance(double speed) const;

private:
    PlatooningParameters platooningParameters;

    std::shared_ptr<Plexe::VEHICLE_DATA> cachedFollowerData;

    Eigen::Matrix2d P0;
    double kfPredecessorTime, kfFollowerTime;
    std::shared_ptr<KalmanFilter> kfPredecessor;
    std::shared_ptr<KalmanFilter> kfFollower;

    std::shared_ptr<RunningAverage<double>> distanceKFAvg;
    std::shared_ptr<RunningAverage<double>> distanceRadarAvg;
    std::shared_ptr<RunningAverage<double>> distanceV2XKFAvg;
    std::shared_ptr<RunningAverage<double>> distanceRadarKFAvg;
    std::shared_ptr<RunningAverage<double>> speedV2XKFAvg;
    std::shared_ptr<RunningAverage<double>> speedRadarV2XAvg;
    std::shared_ptr<RunningAverage<double>> speedRadarKFAvg;
    std::shared_ptr<RunningAverage<double>> speedKFAvg;

    using DistanceDetectorType1 = AttackDetector<std::function<double(double)>>;
    using DistanceDetectorType2 = AttackDetector<std::function<double(double, double)>>;
    using SpeedDetectorType1 = AttackDetector<std::function<double(double, double)>>;
    using SpeedDetectorType2 = AttackDetector<std::function<double(double)>>;
    using SpeedDetectorType3 = AttackDetector<std::function<double(double, double, double)>>;
    /*
    std::shared_ptr<DistanceDetectorType1> distanceKFDetector;
    std::shared_ptr<DistanceDetectorType1> distanceRadarDetector;
    std::shared_ptr<DistanceDetectorType2> distanceV2XKFDetector;
    std::shared_ptr<DistanceDetectorType2> distanceRadarKFDetector;

    std::shared_ptr<SpeedDetectorType1> speedV2XKFDetector;
    std::shared_ptr<SpeedDetectorType2> speedRadarV2XDetector;
    std::shared_ptr<SpeedDetectorType3> speedRadarKFDetector;
    */
    //Machine Learning Detector
    std::shared_ptr<AttackDetectorML> distanceKFDetectorML;
    std::shared_ptr<AttackDetectorML> distanceV2XKFDetectorML;
    std::shared_ptr<AttackDetectorML> distanceRadarDetectorML;
    std::shared_ptr<AttackDetectorML> distanceRadarKFDetectorML;

    std::shared_ptr<AttackDetectorML> speedV2XKFDetectorML;
    std::shared_ptr<AttackDetectorML> speedRadarV2XDetectorML;
    std::shared_ptr<AttackDetectorML> speedRadarKFDetectorML;
    std::shared_ptr<AttackDetectorML> speedKFDetectorML;

};

#endif // BEACONANALYZER_H
