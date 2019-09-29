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

#ifndef ATTACKDETECTOR_H
#define ATTACKDETECTOR_H

template <typename F>
class AttackDetector {

public:
    /**
     * Constructs a new AttackDetector object
     * @param attackTolerance: the number of consecutive values above the threshold necessary to identify an attack.
     */
    AttackDetector(F&& thresholdFunction, std::size_t attackTolerance)
        : thresholdFunction(thresholdFunction)
        , attackTolerance(attackTolerance)
        , attackCounter(0)
    {
    }

    /**
     * Returns whether an attack has been detected.
     */
    bool attackDetected() const
    {
        return attackCounter >= attackTolerance;
    }

    /**
     * Updates and returns whether an attack has been detected.
     * @param value: the value compared to the threshold to detect the attack.
     * @param args: the arguments forwarded to the thresholdFunction.
     */
    template <typename... Args>
    bool update(double value, Args&&... args)
    {

        if (!attackDetected()) {
            if (std::abs(value) < thresholdFunction(args...)) {
                attackCounter = 0;
            }
            else {
                attackCounter++;
            }
        }

        return attackDetected();
    }

private:
    const F thresholdFunction;
    const size_t attackTolerance;
    size_t attackCounter;
};

#endif // ATTACKDETECTOR_H