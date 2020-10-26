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
        , bufferAvg(size, 0)
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
        // buffer[index] = value;
        // index = (index + 1) % buffer.size();
        std::rotate(buffer.begin(), buffer.begin() + 1, buffer.end());
        buffer[buffer.size() - 1] = value;
        elements = elements == buffer.size() ? elements : elements + 1;
    }

    std::vector<T> getBuffer()
    {
        return buffer;
    }

    std::vector<T> getBufferAvg()
    {
        return bufferAvg;
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
     * Returns the running average computed on the stored data
     * @return the computed running average
     */
    std::vector<T> getRunningAverage_mod()
    {
        std::vector<double> w = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
        double sum = 0;
        int i;
        for (i = 0; i < buffer.size() || i < 10; i++) {
            sum += w[i] * buffer[i];
            // std::cout<<buffer[i];
        }
        //std::cout<<"\nW_average"<<sum/5.5<<"i_out"<<i<<std::endl;

        std::rotate(bufferAvg.begin(), bufferAvg.begin() + 1, bufferAvg.end());
        bufferAvg[bufferAvg.size() - 1] = sum/5.5;

        return bufferAvg;
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
    std::vector<T> bufferAvg; 
    std::size_t index; // The index pointing to the next insertion position
    std::size_t elements; // The number of elements stored in the buffer
};

#endif // RUNNINGAVERAGE_H
