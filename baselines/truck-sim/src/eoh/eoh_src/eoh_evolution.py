import re
import time

from ..llm.interface_LLM import InterfaceLLM


class Evolution:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode,prompts):
        # set prompt interface
        self.prompt_task         = prompts.get_task()
        self.prompt_task_ext     = prompts.get_task_ext()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()

        self.joined_inputs = ", ".join(f"'{s}'" for s in self.prompt_func_inputs)
        self.joined_outputs = ", ".join(f"'{s}'" for s in self.prompt_func_outputs)

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode)

    def get_prompt_i1(self):
        prompt_content = (
            f"{self.prompt_task}\n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content

    def get_prompt_i2(self):
        prompt_content = (
            f"{self.prompt_task_ext}\n"
            f"First, describe your algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                f"{prompt_indiv}No.{str(i+1)} algorithm and the corresponding code are: \n"
                f"{indivs[i]['algorithm']}\n"
                f"{indivs[i]['code']}\n"
            )
        # for i in range(len(indivs)):
        #     prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = (
            f"{self.prompt_task}\n"
            f"I have {len(indivs)} existing algorithms with their codes as follows: \n"
            f"{prompt_indiv}"
            f"Please help me create a new algorithm that has a totally different form from the given ones. \n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content
    
    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                f"{prompt_indiv}No.{str(i+1)} algorithm and the corresponding code are: \n"
                f"{indivs[i]['algorithm']}\n"
                f"{indivs[i]['code']}\n"
            )
        # for i in range(len(indivs)):
        #     prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = (
            f"{self.prompt_task}\n"
            f"I have {len(indivs)} existing algorithms with their codes as follows: \n"
            f"{prompt_indiv}"
            f"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
            f"Firstly, identify the common backbone idea in the provided algorithms. "
            f"Secondly, based on the backbone idea describe your new algorithm in one sentence. "
            f"The description must be inside a brace. Thirdly, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content
    
    def get_prompt_m1(self, indiv1):
        prompt_content = (
            f"{self.prompt_task}\n"
            f"I have one algorithm with its code as follows. "
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n"
            f"{indiv1['code']}\n"
            f"Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content
    
    def get_prompt_m2(self, indiv1):
        prompt_content = (
            f"{self.prompt_task}\n"
            f"I have one algorithm with its code as follows. "
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n"
            f"{indiv1['code']}\n"
            f"Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content
    
    def get_prompt_m3(self, indiv1):
        prompt_content = (
            f"First, you need to identify the main components in the function below. "
            f"Next, analyze whether any of these components can be overfit to the in-distribution instances. "
            f"Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
            f"Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"
            f"{indiv1['code']}\n"
            f"{self.prompt_inout_inf}\n"
            f"Do not give additional explanations."
        )

        return prompt_content

    def _get_alg(self, prompt_content):
        for i in range(5):
            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if len(algorithm) == 0 or len(code) == 0:
                if i == 4:
                    print("Error: 5 attempts made")
                else:
                    print(f"Error: on attempt {i}/5 the algorithm or code not identified, waiting 1 second")
                    time.sleep(1)

            else:
                algorithm = algorithm[0]
                code = code[0]
                break

        # return values are replaced with self.prompt_func_outputs
        code_all = f"{code} " + ", ".join(s for s in self.prompt_func_outputs)
        return [code_all, algorithm]


    def i1(self):
        prompt_content = self.get_prompt_i1()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def i2(self):
        prompt_content = self.get_prompt_i2()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self,parents):
        prompt_content = self.get_prompt_e1(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
        prompt_content = self.get_prompt_e2(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
        prompt_content = self.get_prompt_m1(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
        prompt_content = self.get_prompt_m2(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m3(self,parents):
        prompt_content = self.get_prompt_m3(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]