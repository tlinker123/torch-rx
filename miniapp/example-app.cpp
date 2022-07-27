#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
int main() {
  //torch::Tensor tensor = torch::rand({2, 3});
  //std::cout << tensor << std::endl;
  int feature_size=6;
  int ntypes=2;
  torch::jit::script::Module module;
  try {
     //Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("./H.model_scripted.pt");
           }
  catch (const c10::Error& e) {
          std::cerr << "error loading the model\n";
          return -1;
                       }
  std::cout << "ok\n";                    
  std::vector<torch::jit::IValue> inputs;
  auto inputs_tensor=torch::ones({1, feature_size}).requires_grad_(true);
  inputs.push_back(inputs_tensor);
  //auto inputs = (torch::ones({2,feature_size},torch::requires_grad()));
  //at::Tensor output = module.forward(inputs).toTensor();
  auto output = module.forward(inputs);
  std::cout << output << std::endl;
  auto grad_output = torch::ones_like(output.toTensor());
  auto gradient = torch::autograd::grad({output.toTensor()}, {inputs_tensor}, /*grad_outputs=*/{grad_output}, /*create_graph=*/false);
  //output.backward();
  std::cout << gradient <<std::endl;
  std::cout << "ok\n";                    
}
