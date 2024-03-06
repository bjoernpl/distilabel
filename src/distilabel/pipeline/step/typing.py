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

from typing import Any, Dict, Iterator, List, Tuple, TypeVar, Union

from pydantic import Field
from typing_extensions import Annotated

StepInput = Annotated[List[Dict[str, Any]], "StepInput"]
"""StepInput is just an `Annotated` alias of the typing `List[Dict[str, Any]]` with
extra metadata that allows `distilabel` to perform validations over the `process` step
method defined in each `Step`"""

StepOutput = Annotated[Iterator[List[Dict[str, Any]]], "StepOutput"]
"""StepOutput is just an `Annotated` alias of the typing `Iterator[List[Dict[str, Any]]]`"""

GeneratorStepOutput = Annotated[
    Iterator[Tuple[List[Dict[str, Any]], bool]], "GeneratorStepOutput"
]
"""GeneratorStepOutput is just an `Annotated` alias of the typing `Iterator[Tuple[List[Dict[str, Any]], bool]]`"""

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"

RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]
"""Used to mark the attributes of a `Step` as a runtime parameter."""