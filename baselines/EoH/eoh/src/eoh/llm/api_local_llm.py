# This file includes classe to get response from deployed local LLM
import json
from typing import Collection


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, url, max_attempts=5):
        self._url = url  # 'http://127.0.0.1:11045/completions'
        self.max_attempts = max(1, int(max_attempts))

    def get_response(self, content: str):
        for _ in range(self.max_attempts):
            try:
                response = self._do_request(content)
                return response
            except:
                continue
        return None

    def _do_request(self, content: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests to use EoH local LLM mode") from exc

        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        print(response)
        if response.status_code == 200:
            response = response.json()['content'][0]
            return response
