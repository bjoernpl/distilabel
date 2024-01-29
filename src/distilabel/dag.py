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

from collections import defaultdict
from typing import Generator, List

import networkx as nx
from datasets import Dataset

from distilabel.step import Step


class Pipeline:
    def __init__(self) -> None:
        self.dag = nx.DiGraph()
        self.steps = {}

    def add_step(self, step: Step, name: str) -> None:
        if self.dag.has_node(name):
            raise ValueError(f"Step with name '{name}' already exists in the pipeline.")
        if not isinstance(step, Step):
            raise ValueError("`step` must be a subclass of `Step`.")
        self.dag.add_node(name)
        self.steps[name] = step

    def add_edge(self, from_step: str, to_step: str) -> None:
        if not self.dag.has_node(from_step):
            raise ValueError(
                f"Step with name '{from_step}' does not exist in the pipeline."
            )
        if not self.dag.has_node(to_step):
            raise ValueError(
                f"Step with name '{to_step}' does not exist in the pipeline."
            )
        if to_step in nx.ancestors(self.dag, from_step):
            raise ValueError(
                f"Edge from '{from_step}' to '{to_step}' would create a cycle."
            )
        self.dag.add_edge(from_step, to_step)

    def _iter_dag_steps(self) -> Generator[List[str], None, None]:
        """Iterate over steps in the DAG based on their trophic levels. This is similar
        to a topological sort, but we also know which steps are at the same level and
        can be run in parallel.

        Yields:
            A list containing the names of the steps that can be run in parallel.
        """
        trophic_levels = nx.trophic_levels(self.dag)

        v = defaultdict(list)
        for step, trophic_level in trophic_levels.items():
            v[int(trophic_level)].append(step)

        for trophic_level in sorted(v.keys()):
            yield v[trophic_level]

    def run(self, dataset: Dataset) -> Dataset:
        for steps in self._iter_dag_steps():
            print(steps)
