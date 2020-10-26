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

#ifndef ATTACKDETECTORML_H
#define ATTACKDETECTORML_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <fdeep/fdeep.hpp>

class AttackDetectorML {

public:
    /**
     * Constructs a new AttackDetectorML object
     * @param ML_accuracy: the number of consecutive values above the threshold necessary to identify an attack.
     */
    AttackDetectorML(double ML_accuracy, std::string name_value, std::string ML_model_path)
        : ML_accuracy(ML_accuracy)
        , name_value (name_value)
    {
        
        auto model_temp = fdeep::load_model(ML_model_path+"model_" + name_value + ".json", true, fdeep::dev_null_logger);
        model = std::make_unique<fdeep::model>(model_temp);
        //model = std::move(model_temp);

        std::ifstream infile(ML_model_path+"scaler2_" + name_value + ".txt");
        
        infile >> mean_scaler >> std_scaler;
        //std::cout<< "acc: "<< ML_accuracy << "path" << ML_model_path<< std::endl;
        //std::cout<< "mean_scaler: "<< mean_scaler << "std_scaler: " << std_scaler << std::endl;
        isAttack = false;
    }

    /**
     * Returns whether an attack has been detected.
     */
    bool attackDetected() const
    {
        return isAttack;
    }

    /**
     * Updates and returns whether an attack has been detected.
     * @param value: the value compared to the threshold to detect the attack.
     */

    bool update(std::vector<double> value, float simTime)
    {
        //std::vector<double> value = {-0.1641943, 0.00476462, -0.00567606, -0.06135376, -0.04872626, -0.12785812, -0.2246094, -0.24044481, -0.30621219, -0.24153783};
        if(simTime<10)
            return false;
        //std::cout << "Vvalue: " << name_value << " " ;
        if (!attackDetected()) {
            auto print = [](const float& n) { std::cout << " " << n; };

            std::vector<float> floatVec(value.begin(), value.end());

            //std::cout << "\nValue: Before" ;
            //std::for_each(floatVec.cbegin(), floatVec.cend(), print);
            std::for_each(floatVec.begin(), floatVec.end(), [this](float &n){ if(n!=0) n=(n-mean_scaler)/std_scaler; else n = 0; });
            //std::cout << "\nValue: After" ;
            //std::for_each(floatVec.cbegin(), floatVec.cend(), print);
            //std::cout << std::endl;

            const fdeep::tensor t(fdeep::tensor_shape(10, 1), floatVec);
            const auto result = model->predict({t});
            
            /*for (int i = 0; i < floatVec.size(); i++){
                std::cout << floatVec[i] << " ";
            }*/
            
            
            if (result[0].to_vector()[0] < ML_accuracy) {
                isAttack = false;
            }
            else {
                isAttack = true;
                std::cout << " TH:" << ML_accuracy << " RESULT: " << result[0].to_vector()[0] << std::endl;
                std::cout<<"ISTANT DETECTION: " << simTime <<" NV: " << name_value <<std::endl;
            }
        } else{
            std::cout<<"DETECTION: " << simTime <<" NV: " << name_value <<std::endl;
        }

        return attackDetected();
    }

private:
    std::unique_ptr<fdeep::model> model;
    std::string name_value;
    bool isAttack;
    double ML_accuracy;
    float mean_scaler;
    float std_scaler;
};

#endif // AttackDetectorML_H