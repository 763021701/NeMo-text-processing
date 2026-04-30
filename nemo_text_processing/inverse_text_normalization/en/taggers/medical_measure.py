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

"""
Medical unit spoken phrases composed as numerator + (per + denominator)+.

Data files (written TAB spoken, same convention as measurements.tsv):
- data/measurements/medical/numerators.tsv
- data/measurements/medical/denominators.tsv

This compound path is preferred in MeasureFst over string-file rows when both apply.
Non-composable or special forms (e.g. vol %) live in measurements_medical.tsv.

v1 supports:
- Binary:  A per B -> A/B (e.g. millimole per liter -> mmol/L).
- Ternary: A per B per C -> A/B/C (e.g. milligrams per kilogram per day -> mg/kg/day).

MeasureFst may also chain ``unit_singular + unit_misc`` (binary compound + a further
``per`` + unit from the main graph); ternary here keeps three medical segments in one FST.

Output is spoken (lower case in TSV) -> written abbreviation.
"""

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import delete_space


def build_medical_compound_spoken_to_written_fst() -> "pynini.FstLike":
    """
    Returns FST mapping English spoken medical compound units to written form.

    Does not include casing on the outer compose; caller may union with casing
    the same way as base measurements.
    """
    numerators = pynini.string_file(get_abs_path("data/measurements/medical/numerators.tsv"))
    denominators = pynini.string_file(get_abs_path("data/measurements/medical/denominators.tsv"))

    # Do not apply get_singulars to the whole numerator/denominator maps: English plural
    # rules can map biological plurals incorrectly (e.g. cells -> cell). Prefer explicit
    # plural spoken rows in numerators.tsv / denominators.tsv.
    num = pynini.invert(numerators).optimize()
    den = pynini.invert(denominators).optimize()

    per_slash_den = (
        delete_space
        + pynutil.delete("per")
        + delete_space
        + pynutil.insert("/")
        + den
    ).optimize()

    compound_binary = (num + per_slash_den).optimize()
    compound_ternary = (num + per_slash_den + per_slash_den).optimize()

    return pynini.union(compound_binary, compound_ternary).optimize()
