//
// Copyright (c) 2019 Marco Iorio <marco.iorio@polito.it>
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

#include "veins/modules/application/platooning/apps/InjectionDetectionApp.h"
#include "veins/modules/application/platooning/scenarios/BaseScenario.h"
#include "veins/modules/application/platooning/sensors/SensorParameters.h"

using namespace Veins;

Define_Module(InjectionDetectionApp);

InjectionDetectionApp::InjectionDetectionApp()
    : switchedToACC(false)
    , leaderAttackDetectedTime(-1)
    , leaderAttackDetectedType(0)
    , predecessorAttackDetectedTime(-1)
    , predecessorAttackDetectedType(0)
{
}

InjectionDetectionApp::~InjectionDetectionApp() = default;

void InjectionDetectionApp::initialize(int stage)
{
    SimplePlatooningApp::initialize(stage);

    if (stage == 0) {
        fallbackACCHeadway = par("fallbackACCHeadway");
        fallbackACCSafetyMargin = par("fallbackACCSafetyMargin");

        detectionParameters.runningAvgWindow = par("detectionAvgWindow");
        detectionParameters.attackTolerance = par("detectionAttackTolerance");
        detectionParameters.accelerationFactor = par("detectionAccelerationFactor");

        detectionParameters.distanceKFThresholdFactor = par("distanceKFThresholdFactor");
        detectionParameters.distanceRadarThresholdFactor = par("distanceRadarThresholdFactor");
        detectionParameters.distanceV2XKFThresholdFactor = par("distanceV2XKFThresholdFactor");
        detectionParameters.distanceRadarKFThresholdFactor = par("distanceRadarKFThresholdFactor");
        detectionParameters.speedV2XKFThresholdFactor = par("speedV2XKFThresholdFactor");
        detectionParameters.speedRadarV2XThresholdFactor = par("speedRadarV2XThresholdFactor");
        detectionParameters.speedRadarKFThresholdFactor = par("speedRadarKFThresholdFactor");

        detectionParameters.ML_model_path = par("ML_model_path").stdstringValue();
        detectionParameters.ML_accuracy = par("ML_accuracy");

        qFactor = par("qFactor");
    }

    if (stage == 1) {
        std::map<enum Plexe::VEHICLE_SENSORS, SensorParameters*> sensorParameters;
        for (int i = 0; i <= Plexe::VEHICLE_SENSORS::RADAR_SPEED; i++) {
            auto submodule = dynamic_cast<SensorParameters*>(getParentModule()->getSubmodule("sensors", i));
            sensorParameters.emplace(static_cast<Plexe::VEHICLE_SENSORS>(i), submodule);
        }

        auto scenario = FindModule<BaseScenario*>::findSubModule(getParentModule());
        BeaconAnalyzerML::PlatooningParameters platooningParameters{
            .spacing = scenario->getSpacing(),
            .headway = scenario->getHeadway(),
        };
  
        leaderDetection = std::make_shared<BeaconAnalyzerML>(platooningParameters, detectionParameters, sensorParameters, qFactor);
        predecessorDetection = std::make_shared<BeaconAnalyzerML>(platooningParameters, detectionParameters, sensorParameters, qFactor);
    }
}

void InjectionDetectionApp::finish()
{
    recordScalar("LeaderAttackDetected", leaderAttackDetectedTime, "s");
    recordScalar("LeaderAttackDetectedType", leaderAttackDetectedType);
    recordScalar("PredecessorAttackDetected", predecessorAttackDetectedTime, "s");
    recordScalar("PredecessorAttackDetectedType", predecessorAttackDetectedType);

    SimplePlatooningApp::finish();
}

void InjectionDetectionApp::onPlatoonBeacon(const PlatooningBeacon* pb)
{
    // The message comes from the same platoon and the current vehicle is not the leader
    if (positionHelper->isInSamePlatoon(pb->getVehicleId()) && !positionHelper->isLeader()) {

        double now = simTime().dbl();
        auto beaconData = beaconToVehicleData(pb);

        // Data coming from the leader vehicle
        if (pb->getVehicleId() == positionHelper->getLeaderId()) {
            leaderDetection->update(beaconData);

            if (!switchedToACC && leaderDetection->attackDetected()) {
                leaderAttackDetectedTime = now;
                leaderAttackDetectedType = leaderDetection->attackDetectedType();
                switchToACC();
            }
        }

        // Data coming from the direct follower of the leader
        if (positionHelper->getMemberPosition(pb->getVehicleId()) == 1) {
            leaderDetection->cacheFollowerData(beaconData);
        }

        // Data coming from my predecessor
        if (positionHelper->getMemberPosition(pb->getVehicleId()) == positionHelper->getPosition() - 1) {
            auto currentData = std::make_shared<Plexe::VEHICLE_DATA>();
            traciVehicle->getVehicleData(currentData.get(), true /* values obtained from realistic sensors */);

            auto radarReading = std::make_shared<Plexe::RADAR_READING>();
            traciVehicle->getRadarMeasurements(radarReading.get(), true /* values obtained from realistic sensors */);

            predecessorDetection->update(beaconData, currentData, radarReading);

            if (!switchedToACC && predecessorDetection->attackDetected()) {
                predecessorAttackDetectedTime = now;
                predecessorAttackDetectedType = predecessorDetection->attackDetectedType();
                switchToACC();
            }
        }
    }

    SimplePlatooningApp::onPlatoonBeacon(pb);
}

void InjectionDetectionApp::switchToACC()
{

    switchedToACC = true;
    if (fallbackACCHeadway > 0) {
        traciVehicle->degradeToACC(fallbackACCHeadway, fallbackACCSafetyMargin);
    }
}
