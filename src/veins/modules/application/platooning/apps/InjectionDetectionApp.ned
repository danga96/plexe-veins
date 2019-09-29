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

package org.car2x.veins.modules.application.platooning.apps;

import org.car2x.veins.modules.application.platooning.apps.SimplePlatooningApp;

simple InjectionDetectionApp extends SimplePlatooningApp
{
    parameters:
        // The headway configured for the ACC if an attack is detected (ACC fallback disabled if set to a value <= 0)
        double fallbackACCHeadway @unit("s");
        // A safety factor applied to initial decelerations to prevent crashes during the switching phase
        double fallbackACCSafetyMargin;

        // The size of the window used to compute the running averages necessary to detect injection attacks
        int detectionAvgWindow;

        // The amount of values exceeding the threshold required to detect an attack
        int detectionAttackTolerance;

        // The "amount" of the expected distance between two vehicles used as a detection threshold
        // (with the positions estimated through the Kalman Filters)

        // The "amount" of the expected distance between two vehicles used as a detection threshold
        double distanceKFThresholdFactor;
        double distanceRadarThresholdFactor;

        // Correcting factors used to scale the computed detection thresholds
        double distanceV2XKFThresholdFactor = default(1);
        double distanceRadarKFThresholdFactor = default(1);
        double speedV2XKFThresholdFactor = default(1);
        double speedRadarV2XThresholdFactor = default(1);
        double speedRadarKFThresholdFactor = default(1);

        // The "amount" of the acceleration value used as a correcting factor to compute part of the detection threshold
        double detectionAccelerationFactor;

        // The dividing factor used (squared) to construct the Q matrix from the R one
        double qFactor;

        @display("i=block/app2");
        @class(InjectionDetectionApp);
}
