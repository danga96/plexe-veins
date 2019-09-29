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

#ifndef INJECTIONDETECTIONAPP_H_
#define INJECTIONDETECTIONAPP_H_

#include "veins/modules/application/platooning/apps/SimplePlatooningApp.h"
#include "veins/modules/application/platooning/detection/BeaconAnalyzer.h"

class InjectionDetectionApp : public SimplePlatooningApp {

public:
    InjectionDetectionApp();
    ~InjectionDetectionApp() override;

    void initialize(int stage) override;
    void finish() override;

protected:
    void onPlatoonBeacon(const PlatooningBeacon* pb) override;

    void switchToACC();

protected:
    std::shared_ptr<BeaconAnalyzer> leaderDetection;
    std::shared_ptr<BeaconAnalyzer> predecessorDetection;

    bool switchedToACC;
    double leaderAttackDetectedTime;
    double leaderAttackDetectedType;
    double predecessorAttackDetectedTime;
    double predecessorAttackDetectedType;

    BeaconAnalyzer::DetectionParameters detectionParameters;
    double fallbackACCHeadway;
    double fallbackACCSafetyMargin;
    double qFactor;
};

#endif /* INJECTIONDETECTIONAPP_H_ */
