# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from typing import Optional

from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)


class ScientificFst(GraphFst):
    """
    Finite state transducer for classifying scientific notation.
        e.g. one point two three times ten to the minus five ->
            scientific { mantissa: "1.23" exponent: "-5" }

    Exponent may be cardinal (``five``) or ordinal (``fifth``) when ``ordinal`` is provided.

    Attributes:
        final_graph: inner fields ``mantissa`` / ``exponent`` without the outer ``scientific { }``
            wrapper, for composition inside ``measure``.
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        ordinal: Optional[GraphFst] = None,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="scientific", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        mantissa_integer = cardinal_graph
        mantissa_decimal = cardinal_graph + delete_space + pynutil.insert(".") + pynutil.delete("point") + delete_space + decimal.graph
        mantissa_number = mantissa_decimal | mantissa_integer
        optional_mantissa_sign = pynini.closure(pynini.cross(MINUS, "-") + delete_space, 0, 1)

        mantissa = pynutil.insert('mantissa: "') + optional_mantissa_sign + mantissa_number + pynutil.insert('"')

        marker = (
            pynutil.delete("times")
            + delete_space
            + pynutil.delete("ten")
            + delete_space
            + (
                pynutil.delete("to the power of")
                | pynutil.delete("raised to the")
                | pynutil.delete("to the")
            )
        )

        optional_exponent_sign = pynini.closure(pynini.cross(MINUS, "-") + delete_space, 0, 1)
        exponent_value = cardinal_graph
        if ordinal is not None:
            exponent_value |= ordinal.graph
        exponent = pynutil.insert('exponent: "') + optional_exponent_sign + exponent_value + pynutil.insert('"')

        inner = mantissa + delete_extra_space + marker + delete_extra_space + exponent

        if input_case == INPUT_CASED:
            inner = capitalized_input_graph(inner)

        self.final_graph = inner.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
