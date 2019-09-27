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

#ifndef SENSORPARAMETERS_H_
#define SENSORPARAMETERS_H_

#include "veins/base/modules/BaseModule.h"
#include "veins/modules/application/platooning/CC_Const.h"

class SensorParameters : public Veins::BaseModule {

public:
    SensorParameters() = default;
    ~SensorParameters() override = default;

    void initialize(int stage) override;

    Plexe::VEHICLE_SENSORS getSensorType() const
    {
        return sensorType;
    }
    double getMinValue() const
    {
        return minValue;
    }
    double getMaxValue() const
    {
        return maxValue;
    }
    int getDecimalDigits() const
    {
        return decimalDigits;
    }
    double getUpdateInterval() const
    {
        return updateInterval;
    }
    double getAbsoluteError() const
    {
        return absoluteError;
    }
    double getPercentageError() const
    {
        return percentageError;
    }
    bool getSumErrors() const
    {
        return sumErrors;
    }
    int getSeed() const
    {
        return seed;
    }

private:
    Plexe::VEHICLE_SENSORS sensorType;
    double minValue;
    double maxValue;
    int decimalDigits;
    double updateInterval;
    double absoluteError;
    double percentageError;
    bool sumErrors;
    int seed;
};

#endif /* SENSORPARAMETERS_H_ */
