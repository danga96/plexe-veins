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

#include "veins/modules/application/platooning/scenarios/RandomScenario.h"

#define HIGH_FOLLOWER_DESIRED_SPEED 1000 /* m/s */

Define_Module(RandomScenario);

RandomScenario::RandomScenario() = default;

RandomScenario::~RandomScenario()
{
    cancelAndDelete(changeSpeed);
    changeSpeed = nullptr;
}

void RandomScenario::initialize(int stage)
{
    BaseScenario::initialize(stage);

    if (stage == 1) {

        // The current vehicle is a leader, and thus the speed profile is configured
        if (positionHelper->getId() < positionHelper->getLanesCount()) {

            // Read the different configuration parameters
            avgSpeed = par("avgSpeed");
            maxSpeed = par("maxSpeed");

            if (avgSpeed > maxSpeed) {
                avgSpeed = maxSpeed;
            }

            minAcceleration = par("minAcceleration");
            maxAcceleration = par("maxAcceleration");
            avgAcceleration = par("avgAcceleration");
            minDeceleration = par("minDeceleration");
            maxDeceleration = par("maxDeceleration");
            avgDeceleration = par("avgDeceleration");

            accelerationProbability = par("accelerationProbability");
            decelerationProbability = par("decelerationProbability");

            startTime = SimTime(par("startTime"));
            minStepDuration = SimTime(par("minStepDuration"));
            meanStepDuration = SimTime(par("meanStepDuration"));

            // Configure the initial speed
            traciVehicle->setCruiseControlDesiredSpeed(avgSpeed);

            // Schedule the first acceleration change
            changeSpeed = new cMessage("changeSpeed");
            scheduleAt(simTime() > startTime ? simTime() : startTime, changeSpeed);
        }
        else {
            // let the follower set a very high desired speed to stay connected
            // to the leader when it is accelerating
            traciVehicle->setCruiseControlDesiredSpeed(HIGH_FOLLOWER_DESIRED_SPEED);
        }
    }
}

void RandomScenario::handleSelfMsg(cMessage* msg)
{
    if (msg == changeSpeed) {
        Plexe::VEHICLE_DATA vehicleData{};
        traciVehicle->getVehicleData(&vehicleData, false);

        // Set a new fixed acceleration value and schedule the next change
        double acceleration = drawAcceleration(vehicleData.speed);
        SimTime duration = drawDuration(vehicleData.speed, acceleration);

        traciVehicle->setFixedAcceleration(1, acceleration);
        scheduleAt(simTime() + duration, changeSpeed);
    }
    else {
        BaseScenario::handleSelfMsg(msg);
    }
}

double RandomScenario::drawAcceleration(double currentSpeed) const
{
    double choice = uniform(0, 1);

    double correctedAccelerationProbability = currentSpeed < avgSpeed ? accelerationProbability : accelerationProbability * (maxSpeed - currentSpeed) / (maxSpeed - avgSpeed);
    double correctedDecelerationProbability = currentSpeed > 0.25 * avgSpeed ? decelerationProbability : decelerationProbability * (currentSpeed) / (0.25 * avgSpeed);

    // Acceleration
    if (choice < correctedAccelerationProbability) {
        double acceleration = exponential(avgAcceleration);
        return std::min(std::max(minAcceleration, acceleration), maxAcceleration);
    }

    // Deceleration
    if (choice < correctedAccelerationProbability + correctedDecelerationProbability) {
        double deceleration = exponential(avgDeceleration);
        return -1 * std::min(std::max(minDeceleration, deceleration), maxDeceleration);
    }

    // No speed changes
    return 0;
}

SimTime RandomScenario::drawDuration(double currentSpeed, double currentAcceleration) const
{
    SimTime duration = minStepDuration + exponential(meanStepDuration);
    SimTime durationToMaxSpeed = std::max(0.0, (maxSpeed - currentSpeed) / maxAcceleration);
    return currentAcceleration > 0 ? std::min(duration, durationToMaxSpeed) : duration;
}