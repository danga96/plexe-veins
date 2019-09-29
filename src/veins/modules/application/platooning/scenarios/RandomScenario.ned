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

package org.car2x.veins.modules.application.platooning.scenarios;

import org.car2x.veins.modules.application.platooning.scenarios.BBaseScenario;

simple RandomScenario extends BBaseScenario
{
    parameters:
        double avgSpeed @unit("mps");
        double maxSpeed @unit("mps");

        double minAcceleration @unit("mpsps") = default(0.1mpsps);
        double maxAcceleration @unit("mpsps") = default(2.0mpsps);
        double avgAcceleration @unit("mpsps");
        double minDeceleration @unit("mpsps") = default(0.1mpsps);
        double maxDeceleration @unit("mpsps") = default(4.0mpsps);
        double avgDeceleration @unit("mpsps");

        double accelerationProbability;
        double decelerationProbability;

        double startTime @unit("s");
        double minStepDuration @unit("s");
        double meanStepDuration @unit("s");

        @display("i=block/app2");
        @class(RandomScenario);
}
