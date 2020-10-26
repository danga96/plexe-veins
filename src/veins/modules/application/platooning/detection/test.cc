#include <fdeep/fdeep.hpp>
#include <bits/stdc++.h> 

int main()
{
    clock_t start, end; 
    const fdeep::model model = fdeep::load_model("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_V2XKFdistance.json", true, fdeep::dev_null_logger);
    const std::vector<float> v = {-0.1641943, 0.00476462, -0.00567606, -0.06135376, -0.04872626, -0.12785812, -0.2246094, -0.24044481, -0.30621219, -0.24153783};
    
    start = clock();
    const fdeep::tensor t(fdeep::tensor_shape(10, 1), v);
    const auto result = model.predict({t});
    //std::cout <<"Value: " << result.at(0) << std::endl;
    //std::cout << fdeep::show_tensors(result.at(0)) << std::endl;
    //std::vector<float> vec = result[0].to_vector();
    std::cout << "Value: " << result[0].to_vector()[0] << std::endl;
    end = clock();

    std::cout << "Time taken by program is : " << std::fixed  
         << (end-start)/double(CLOCKS_PER_SEC) << std::setprecision(5); 
    std::cout << " sec " << std::endl; 
}
