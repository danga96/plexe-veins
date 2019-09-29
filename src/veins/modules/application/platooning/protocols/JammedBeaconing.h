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

#ifndef JAMMEDBEACONING_H_
#define JAMMEDBEACONING_H_

#include "SimplePlatooningBeaconing.h"

class JammedBeaconing : public SimplePlatooningBeaconing {
public:
    JammedBeaconing();
    ~JammedBeaconing() override = default;

    void initialize(int stage) override;

protected:
    void sendPlatooningMessage(int destinationAddress) override;
    void handleLowerMsg(cMessage* msg) override;
    void handleUpperMsg(cMessage* msg) override;

    bool underJammingAttack();

private:
    double attackStart;
    double attackStop;
};

#endif /* JAMMEDBEACONING_H_ */
