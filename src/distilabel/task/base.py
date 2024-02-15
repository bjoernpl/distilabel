# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

from distilabel.llm import LLM
from distilabel.step.base import Step, StepInput

ChatType = List[Dict[Literal["role", "content"], str]]


class TaskStep(Step, ABC):
    llm: "LLM"

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        pass

    def process(self, inputs: StepInput) -> StepInput:
        results = []
        for input in inputs:
            formatted_input = self.format_input(input)
            output = self.llm.generate(formatted_input)  # type: ignore
            formatted_output = self.format_output(output)  # type: ignore
            results.append({**input, **formatted_output})
        return results


def is_openai_format(input: Any) -> bool:
    if isinstance(input, list):
        if all(
            isinstance(item, dict)
            and "role" in item
            and "content" in item
            and isinstance(item["role"], str)
            and isinstance(item["content"], str)
            for item in input
        ):
            return True
    return False


class TextGenerationStep(TaskStep):
    inputs: List[str]
    outputs: List[str]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        if len(self.inputs) != 1:
            raise ValueError(
                "Only one input argument is supported by default. If you are"
                " willing to use more, feel free to extend this class and override"
                " the `format_input` method."
            )
        input = input[self.inputs[0]]
        if isinstance(input, str):
            return [
                {"role": "system", "content": ""},
                {"role": "user", "content": input},
            ]
        if is_openai_format(input=input):
            return input  # type: ignore
        raise ValueError(
            "Supported formats are `str` and `ChatType`. If you are willing"
            " to use other type, feel free to extend this class and override"
            " the `format_input` method."
        )

    def format_output(self, output: str) -> Dict[str, Any]:
        return {"generations": output}


# if __name__ == "__main__":
#     task = TextGenerationStep(
#         name="text-generation",
#         inputs=["question"],
#         outputs=["generations"],
#         llm=LLM(),
#     )
