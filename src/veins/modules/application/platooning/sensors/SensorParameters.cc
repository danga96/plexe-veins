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

#include "veins/modules/application/platooning/sensors/SensorParameters.h"

#include "veins/modules/mobility/traci/TraCIMobility.h"
#include "veins/modules/application/platooning/CC_Const.h"

using namespace Veins;

Define_Module(SensorParameters);

void SensorParameters::initialize(int stage)
{
    BaseModule::initialize(stage);

    if (stage == 0) {
        sensorType = static_cast<Plexe::VEHICLE_SENSORS>(par("sensorType").intValue());

        minValue = par("minValue");
        maxValue = par("maxValue");
        decimalDigits = par("decimalDigits");
        updateInterval = par("updateInterval");

        absoluteError = par("absoluteError");
        percentageError = par("percentageError");
        sumErrors = par("sumErrors");
        seed = par("seed");
    }

    if (stage == 1) {
        Veins::TraCIMobility* mobility = Veins::TraCIMobilityAccess().get(getParentModule());
        ASSERT(mobility);
        Veins::TraCICommandInterface* traci = mobility->getCommandInterface();
        ASSERT(traci);
        Veins::TraCICommandInterface::Vehicle* traciVehicle = mobility->getVehicleCommandInterface();
        ASSERT(traciVehicle);

        traciVehicle->setSensorParametersRange(sensorType, minValue, maxValue, decimalDigits, updateInterval);
        traciVehicle->setSensorParametersErrors(sensorType, absoluteError, percentageError, sumErrors, seed);
    }
}
