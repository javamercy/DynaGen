from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM

class InterfaceLLM:
    def __init__(self, api_endpoint, api_key, model_LLM, *args,
                 health_check=True,max_api_attempts=5):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        if len(args) == 1:
            llm_use_local = False
            llm_local_url = None
            debug_mode = args[0]
        elif len(args) == 3:
            llm_use_local, llm_local_url, debug_mode = args
        else:
            raise TypeError("InterfaceLLM expects debug_mode or llm_use_local, llm_local_url, debug_mode")
        self.debug_mode = debug_mode
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url
        self.health_check = health_check
        self.max_api_attempts = int(max_api_attempts)
        self.total_api_calls = 0
        self.candidate_generation_calls = 0
        self.health_check_calls = 0

        print("- check LLM API")

        if self.llm_use_local:
            print('local llm delopyment is used ...')
            
            if self.llm_local_url == None or self.llm_local_url == 'xxx' :
                print(">> Stop with empty url for local llm !")
                exit()

            self.interface_llm = InterfaceLocalLLM(
                self.llm_local_url,
                max_attempts=self.max_api_attempts
            )

        else:
            print('remote llm api is used ...')

            if self.api_key == None or self.api_endpoint ==None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
                print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
                exit()

            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
                max_attempts=self.max_api_attempts,
            )

            
        if not self.health_check:
            return

        res = self.get_response("1+1=?", purpose="health_check")

        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

        # choose LLMs
        # if self.type == "API2D-GPT":
        #     self.interface_llm = InterfaceAPI2D(self.key,self.model_LLM,self.debug_mode)
        # else:
        #     print(">>> Wrong LLM type, only API2D-GPT is available! \n")

    def get_response(self, prompt_content, purpose="candidate_generation"):
        self.total_api_calls += 1
        if purpose == "candidate_generation":
            self.candidate_generation_calls += 1
        elif purpose == "health_check":
            self.health_check_calls += 1
        response = self.interface_llm.get_response(prompt_content)

        return response

    def get_call_summary(self):
        return {
            "model": self.model_LLM,
            "api_endpoint": self.api_endpoint,
            "llm_use_local": self.llm_use_local,
            "candidate_generation_calls": self.candidate_generation_calls,
            "total_api_calls": self.total_api_calls,
            "health_check_calls": self.health_check_calls,
            "health_check_enabled": self.health_check,
            "max_api_attempts": self.max_api_attempts,
        }
