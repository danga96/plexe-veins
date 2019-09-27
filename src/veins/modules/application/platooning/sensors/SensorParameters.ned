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

package org.car2x.veins.modules.application.platooning.sensors;

simple SensorParameters
{
    parameters:
        // The identifier of the current sensor (according to the Plexe::VEHICLE_SENSORS enum)
        int sensorType;

        // The specification about the minimum and maximum
        // values that can be measured by the sensor
        double minValue = default(-1e25);
        double maxValue = default(1e25);
        // The number of decimal digits of the measurement provided by the sensor
        int decimalDigits = default(5);
        // The amount of time elapsed between two measurements
        double updateInterval @unit("s") = default(0s);

        // The absolute error associated to the sensor
        double absoluteError = default(0);
        // The relative error associated to the sensor (in percentage)
        double percentageError = default(0);
        // Whether absolute and relative errors are summed or only the highest one is applied
        bool sumErrors = default(false);
        // The seed used to initialize the random generator
        int seed = default(0);

        @class(SensorParameters);
}
