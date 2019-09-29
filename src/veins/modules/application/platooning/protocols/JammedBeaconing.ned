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

package org.car2x.veins.modules.application.platooning.protocols;

import org.car2x.veins.modules.application.platooning.protocols.SimplePlatooningBeaconing;

//
// Define the parameters to customize the behavior of the jamming attack
//
simple JammedBeaconing extends SimplePlatooningBeaconing
{
    parameters:
        // The time instant when the jamming attack starts
        double attackStart @unit("s");
        // The time instant when the jamming attack stops
        // The attack never stops if attackStop < attackStart
        double attackStop @unit("s") = default(-1s);

        @class(JammedBeaconing);
}