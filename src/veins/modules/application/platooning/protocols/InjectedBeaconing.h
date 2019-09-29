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

#ifndef INJECTEDBEACONING_H_
#define INJECTEDBEACONING_H_

#include "SimplePlatooningBeaconing.h"
#include <functional>

class InjectedBeaconing : public SimplePlatooningBeaconing {

protected:
    struct InjectionParams {
        bool enabled;

        // Whether the injection is incremental or random values are extracted
        bool injectionRandom;

        // How much the shift value is incremented every second
        double injectionRate;

        // The maximum shift that can be applied to the real value
        double injectionLimit;
    };

public:
    InjectedBeaconing();
    ~InjectedBeaconing() override;

    void initialize(int stage) override;
    void finish() override;

protected:
    PlatooningBeacon* generatePlatooningBeacon() override;

    double computeTimeInjectedValue(double value) const;
    double computeInjectedValue(double value, const InjectionParams& injectionData) const;
    double computeInjectedValueFunction(double value, double& save, const std::function<double(double)>& fn) const;

protected:
    double attackStart;
    double attackStop;

    double timeInjectionRange;
    InjectionParams accelerationInjection;
    InjectionParams positionInjection;
    InjectionParams speedInjection;

    bool coordinatedAttack;
    double previousTime;
    double previousPositionX;
    double previousSpeed;
    double previousSpeedX;
};

#endif /* INJECTEDBEACONING_H_ */
