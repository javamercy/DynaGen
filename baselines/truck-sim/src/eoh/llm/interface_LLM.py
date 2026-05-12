from .api_general import InterfaceAPI


class InterfaceLLM:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

        print("- check remote LLM API")

        if self.api_key is None or self.api_endpoint is None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
            print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
            exit()

        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode)

        res = self.interface_llm.get_response("1+1=?")
        if res is None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def get_response(self, prompt_content):
        return self.interface_llm.get_response(prompt_content)
