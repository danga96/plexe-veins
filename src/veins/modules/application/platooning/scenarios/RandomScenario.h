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

#ifndef RANDOMSCENARIO_H_
#define RANDOMSCENARIO_H_

#include "veins/modules/application/platooning/scenarios/BaseScenario.h"

class RandomScenario : public BaseScenario {

public:
    RandomScenario();
    ~RandomScenario() override;

    void initialize(int stage) override;

protected:
    void handleSelfMsg(cMessage* msg) override;

    double drawAcceleration(double currentSpeed) const;
    SimTime drawDuration(double currentSpeed, double currentAcceleration) const;

protected:
    double avgSpeed;
    double maxSpeed;

    double minDeceleration;
    double maxDeceleration;
    double avgDeceleration;
    double minAcceleration;
    double maxAcceleration;
    double avgAcceleration;

    double accelerationProbability;
    double decelerationProbability;

    SimTime startTime;
    SimTime minStepDuration;
    SimTime meanStepDuration;

    // Message used to change the desired speed of the leader
    cMessage* changeSpeed;
};

#endif // RANDOMSCENARIO_H_
