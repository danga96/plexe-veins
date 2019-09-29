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

#include "InjectedBeaconing.h"

Define_Module(InjectedBeaconing)

InjectedBeaconing::InjectedBeaconing() = default;
InjectedBeaconing::~InjectedBeaconing() = default;

void InjectedBeaconing::initialize(int stage)
{
    SimplePlatooningBeaconing::initialize(stage);

    if (stage == 0) {
        attackStart = par("attackStart");
        attackStop = par("attackStop");

        timeInjectionRange = par("timeInjectionRange");

        positionInjection.enabled = par("enablePositionInjection");
        positionInjection.injectionRandom = par("positionInjectionRandom");
        positionInjection.injectionRate = par("positionInjectionRate");
        positionInjection.injectionLimit = par("positionInjectionLimit");

        speedInjection.enabled = par("enableSpeedInjection");
        speedInjection.injectionRandom = par("speedInjectionRandom");
        speedInjection.injectionRate = par("speedInjectionRate");
        speedInjection.injectionLimit = par("speedInjectionLimit");

        accelerationInjection.enabled = par("enableAccelerationInjection");
        accelerationInjection.injectionRandom = par("accelerationInjectionRandom");
        accelerationInjection.injectionRate = par("accelerationInjectionRate");
        accelerationInjection.injectionLimit = par("accelerationInjectionLimit");

        coordinatedAttack = par("coordinatedAttack");
    }
}

void InjectedBeaconing::finish()
{
    recordScalar("AttackStart", attackStart, "s");
    BaseLayer::finish();
}

PlatooningBeacon* InjectedBeaconing::generatePlatooningBeacon()
{
    auto beacon = SimplePlatooningBeaconing::generatePlatooningBeacon();

    beacon->setTime(computeTimeInjectedValue(beacon->getTime()));
    beacon->setAcceleration(computeInjectedValue(beacon->getAcceleration(), accelerationInjection));
    beacon->setControllerAcceleration(computeInjectedValue(beacon->getControllerAcceleration(), accelerationInjection));

    if (coordinatedAttack) {
        // Inject both speed and position in a coordinated way with respect to acceleration
        double acceleration = beacon->getAcceleration();
        beacon->setPositionX(computeInjectedValueFunction(beacon->getPositionX(), previousPositionX,
                [this, acceleration](double dt) { return previousPositionX + dt * previousSpeed + 0.5 * dt * dt * acceleration; }));
        beacon->setSpeedX(computeInjectedValueFunction(beacon->getSpeedX(), previousSpeedX,
                [this, acceleration](double dt) { return previousSpeedX + dt * acceleration; }));
        beacon->setSpeed(computeInjectedValueFunction(beacon->getSpeed(), previousSpeed,
                [this, acceleration](double dt) { return previousSpeed + dt * acceleration; }));
        previousTime = simTime().dbl();
    }

    else {
        beacon->setPositionX(computeInjectedValue(beacon->getPositionX(), positionInjection));
        beacon->setSpeed(computeInjectedValue(beacon->getSpeed(), speedInjection));
        beacon->setSpeedX(computeInjectedValue(beacon->getSpeedX(), speedInjection));
    }

    return beacon;
}

double InjectedBeaconing::computeTimeInjectedValue(double value) const
{
    double now = simTime().dbl();
    bool underInjectionAttack = now > attackStart && (attackStop < attackStart || now < attackStop);

    if (!underInjectionAttack || timeInjectionRange == 0) {
        return value;
    }

    return uniform(value - timeInjectionRange, value + timeInjectionRange);
}

double InjectedBeaconing::computeInjectedValue(double value, const InjectedBeaconing::InjectionParams& injectionParams) const
{
    double now = simTime().dbl();
    bool underInjectionAttack = now > attackStart && (attackStop < attackStart || now < attackStop);

    if (!underInjectionAttack || !injectionParams.enabled) {
        return value;
    }

    double shift = injectionParams.injectionRate * (now - attackStart);
    shift = (shift < injectionParams.injectionLimit ? shift : injectionParams.injectionLimit);

    return injectionParams.injectionRandom ? uniform(value - shift, value + shift) : value + shift;
}

double InjectedBeaconing::computeInjectedValueFunction(double value, double& save, const std::function<double(double)>& fn) const
{
    double now = simTime().dbl();
    bool underInjectionAttack = now > attackStart && (attackStop < attackStart || now < attackStop);

    if (!underInjectionAttack || !accelerationInjection.enabled) {
        return save = value;
    }

    return save = fn(now - previousTime);
}
