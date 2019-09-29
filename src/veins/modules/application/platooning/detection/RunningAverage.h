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

#ifndef RUNNINGAVERAGE_H
#define RUNNINGAVERAGE_H

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class RunningAverage {
public:
    /**
     * Initializes a new RunningAverage object containing the specified number of elements
     * @param size: the number of values used to compute the average
     */
    explicit RunningAverage(std::size_t size)
        : buffer(size, 0)
        , index(0)
        , elements(0)
    {
    }

    /**
     * Stores a new value into the underlying data structure (replacing an old one if necessary)
     * @param value: the new value to be stored
     */
    void addValue(T value)
    {
        buffer[index] = value;
        index = (index + 1) % buffer.size();
        elements = elements == buffer.size() ? elements : elements + 1;
    }

    /**
     * Returns the running average computed on the stored data
     * @return the computed running average
     */
    T getRunningAverage() const
    {
        return elements == 0 ? 0.0 : std::accumulate(buffer.begin(), buffer.end(), 0.0) / elements;
    }

    /**
     * Returns the running average computed on a subset of the stored data
     * @param subset_size: the number of values belonging to the subset
     * @return the computed running average
     */
    T getRunningAverage(std::size_t subsetSize) const
    {
        if (subsetSize > buffer.size()) {
            throw std::invalid_argument("Specified subset greater than total size");
        }

        std::size_t currentSize = std::min(subsetSize, elements);
        auto begin = buffer.begin() + ((buffer.size() + index - currentSize) % buffer.size());
        auto end = buffer.begin() + index;
        return elements == 0 || subsetSize == 0 ? 0.0 : std::accumulate(begin, end, 0.0) / elements;
    }

private:
    std::vector<T> buffer; // The buffer where the values are stored
    std::size_t index; // The index pointing to the next insertion position
    std::size_t elements; // The number of elements stored in the buffer
};

#endif // RUNNINGAVERAGE_H
