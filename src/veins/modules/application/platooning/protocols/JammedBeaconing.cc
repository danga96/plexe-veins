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

#include "JammedBeaconing.h"

Define_Module(JammedBeaconing)

    JammedBeaconing::JammedBeaconing()
    : attackStart(0)
    , attackStop(0)
{
}

void JammedBeaconing::initialize(int stage)
{
    SimplePlatooningBeaconing::initialize(stage);

    if (stage == 0) {
        attackStart = par("attackStart");
        attackStop = par("attackStop");
    }
}

void JammedBeaconing::sendPlatooningMessage(int destinationAddress)
{
    if (!underJammingAttack()) {
        SimplePlatooningBeaconing::sendPlatooningMessage(destinationAddress);
    }
}

void JammedBeaconing::handleLowerMsg(cMessage* msg)
{
    if (!underJammingAttack()) {
        SimplePlatooningBeaconing::handleLowerMsg(msg);
    }
    else {
        delete msg;
    }
}

void JammedBeaconing::handleUpperMsg(cMessage* msg)
{
    if (!underJammingAttack()) {
        SimplePlatooningBeaconing::handleUpperMsg(msg);
    }
    else {
        delete msg;
    }
}

bool JammedBeaconing::underJammingAttack()
{
    double now = simTime().dbl();
    return now > attackStart && (attackStop < attackStart || now < attackStop);
}
