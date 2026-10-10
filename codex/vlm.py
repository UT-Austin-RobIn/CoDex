import os
import copy
import json
from openai import OpenAI
from PIL import Image
from datetime import datetime
import base64
from io import BytesIO
from typing import List, Optional, Type
from codex.utils.utils_path import PROJECT_ROOT_DIR

OPENAI_REQUEST_TIMEOUT_S = 120.0


class GPT:
    """
    A class to interact with the OpenAI GPT API.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        """
        Initializes the GPT client.

        Reads the API key from the OPENAI_API_KEY environment variable, and the
        model name from CODEX_GPT_MODEL (default "gpt-6-sol").
        """
        from codex.utils.api_keys import get_key
        api_key = get_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or save it in the app "
                "(python scripts/app.py, API keys)."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("CODEX_GPT_MODEL", "gpt-6-sol")
        self.usage_history = []

    def query(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        response_format: Optional[Type] = None
    ) -> str:
        """
        Queries the GPT model with the given text prompt and optional images.

        Args:
            text_prompt (str): The prompt to query the language model.
            images (Optional[List[Image.Image]]): The list of images to query alongside the prompt.
            response_format (Optional[Type]): Expected structured response format.

        Returns:
            str: The generated output from the GPT model.
        """
        # Encode images as base64 strings if provided
        encoded_images = []
        if images:
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                encoded_images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})

        # Prepare messages for the OpenAI API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    *encoded_images,
                ],
            }
        ]
        
        if response_format is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                timeout=OPENAI_REQUEST_TIMEOUT_S,
            )
        else:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                timeout=OPENAI_REQUEST_TIMEOUT_S,
            )
        self.usage_history.append({
            "response_id": response.id,
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None,
        })
        return response.choices[0].message.content


class VLMInterface:
    """
    Lazily constructed access to the GPT vision-language model.
    """

    def __init__(self) -> None:
        """
        Initializes the VLMInterface to manage API connections to models.

        Clients are constructed lazily on first use, so importing/instantiating
        this class requires no API keys. Only the models you actually query
        need their key set (OPENAI_API_KEY).

        """
        self._gpt: Optional[GPT] = None

    @property
    def gpt(self) -> GPT:
        if self._gpt is None:
            self._gpt = GPT()
        return self._gpt


    def query_gpt(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        response_format: Optional[Type] = None
    ) -> str:
        """
        Queries the GPT handler directly.

        Args:
            prompt (str): Prompt string.
            images (Optional[List[Image.Image]]): Selected images.
            response_format (Optional[Type]): Desired structured output type.

        Returns:
            str: API response from GPT.
        """
        return self.gpt.query(prompt, images, response_format=response_format)
    


class VLMSession:
    """
    Manages state and history caching for conversational queries with a VLMInterface.
    Allows for logging requests/responses, resuscitating history contexts, 
    and picking which specific client to invoke on every chat iteration.
    """

    def __init__(
        self,
        interface: VLMInterface,
        log_enabled: bool = True,
        log_dir: str = os.path.join(PROJECT_ROOT_DIR, "out", "vlm_session")
    ) -> None:
        """
        Initializes the VLMSession to store logging states.

        Args:
            interface (VLMInterface): The VLM interface to use for queries.
            log_enabled (bool): Whether to save dialog history to disk.
            log_dir (str): Directory where log JSON files are written.
        """
        self.interface = interface
        self.log_enabled = log_enabled
        self.log_dir = log_dir
        self.reset_history()

    def add_entry(self, prompt: str, response: str) -> None:
        """
        Records an interaction trace in memory.

        Args:
            prompt (str): Text prompt evaluated.
            response (str): System string generated.
        """
        self.history.append({"prompt": prompt, "response": response})

    def get_history_text(self) -> str:
        """
        Consolidates active history logs into context sequences parseable as prompts.

        Returns:
            str: Flat dialog textual representation.
        """
        return "\n".join([f"User: {entry['prompt']}\nResponse: {entry['response']}" for entry in self.history])

    def save_to_log(self) -> None:
        """
        Saves the active session context trace to local JSON logs.
        """
        if self.log_enabled and self.log_file:
            with open(self.log_file, "w") as file:
                json.dump(self.history, file, indent=4)
            gpt = getattr(self.interface, "_gpt", None)
            if gpt is not None and getattr(gpt, "usage_history", None):
                usage_file = os.path.splitext(self.log_file)[0] + "_usage.json"
                with open(usage_file, "w") as file:
                    json.dump(gpt.usage_history, file, indent=2)

    def load_from_log(self, log_file: str) -> None:
        """
        Loads user-agent interaction history from disk storage.

        Args:
            log_file (str): Existing path to target log JSON object.
        """
        with open(log_file, "r") as file:
            self.history = json.load(file)

    def query(
        self,
        images: Optional[List[Image.Image]],
        text_prompt: str,
        include_history: bool = False,
        client: str = "gpt",
        response_format: Optional[Type] = None,
        record_response: bool = True
    ) -> str:
        """
        Pipes prompts to the target VLM API, while coordinating tracking parameters.

        Args:
            images (Optional[List[Image.Image]]): Vision content targets.
            text_prompt (str): Natural language context.
            include_history (bool): Prepend the logged history buffer context trace dynamically.
            client (str): Model handler; only "gpt".
            response_format (Optional[Type]): Valid schema declaration to force structure schema outputs under GPT calls.
            record_response (bool): If active session dict stores these returned sequences internally locally afterwards.

        Returns:
            str: Agent payload response string body.

        Raises:
            ValueError: if invalid string value passes for the `client` target identifier.
        """
        # Include history if requested
        full_prompt = text_prompt
        if include_history:
            history_text = self.get_history_text()
            full_prompt = f"{history_text}\nUser: {text_prompt}"

        # Choose the model to query
        if client != "gpt":
            raise ValueError("Invalid client; only 'gpt' is supported.")
        generated_text = self.interface.query_gpt(full_prompt, images, response_format)

        # Append to chat history
        if record_response:
            self.add_entry(text_prompt, generated_text)

        # Log input and output if logging is enabled
        if self.log_enabled:
            self.save_to_log()

        return generated_text
    
    def dummy_query(
        self,
        images: Optional[List[Image.Image]],
        text_prompt: str,
        response: str,
        include_history: bool = False,
        use_gpt: bool = True
    ) -> str:
        """
        By-passes computation paths entirely simply manually appending an expected prompt 
        reaction and triggering target state logging cycles manually instead of requesting servers.

        Args:
            images (Optional[List[Image.Image]]): Ignored; present for API compatibility.
            text_prompt (str): The prompt to record in history.
            response (str): The predefined response to store and return.
            include_history (bool): Unused; present for API compatibility.
            use_gpt (bool): Unused; present for API compatibility.

        Returns:
            str: The same response string passed in.
        """
        # Append to chat history
        self.add_entry(text_prompt, response)

        # Log input and output if logging is enabled
        if self.log_enabled:
            self.save_to_log()

        return response

    def copy(self) -> 'VLMSession':
        """
        Returns a deep copy of this session with the same history and settings.

        Returns:
            VLMSession: Independent copy of this session.
        """
        new_session = VLMSession(interface=self.interface, log_enabled=self.log_enabled, log_dir=self.log_dir)
        new_session.history = copy.deepcopy(self.history)
        return new_session

    def reset_history(self) -> None:
        """
        Clears conversation history and creates a new timestamped log file.
        """
        self.history = []
        self.log_file = None
        if self.log_enabled and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file = os.path.join(self.log_dir, f"vlm_log_{timestamp}.json")
