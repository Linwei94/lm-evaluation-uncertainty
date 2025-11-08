import re
import sys
import unicodedata

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

import numpy as np

@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps


@register_filter("regex_pos")
class POSFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags if pos_tags else self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        filtered_resps = map(lambda x: filter_set(x), resps)

        return filtered_resps


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading whitespace from responses."""

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp = resp.lstrip()
                filtered_resp.append(resp)
            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex: r's*([A-?])', where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


@register_filter("multi_choice_vnc_regex")
class MultiChoiceVNCRegexFilter(RegexFilter):
    """
    Extracts both the answer letter and the confidence value (0-100)
    from model responses of the form:
        "Answer and Confidence (0-100): A, 92%"
    """

    def __init__(
        self,
        answer_pattern: str = r"([A-Z])",
        confidence_pattern: str = r"(\d{1,3})\s*%?",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(answer_pattern, group_select, fallback)
        self.confidence_pattern = confidence_pattern
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore


    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[tuple[str, float]]]:
        """
        Returns list of lists of (answer_letter, confidence_value)
        """
        resps = list(resps)

        # Normalize inputs: ensure each element is a list of strings
        normalized = []
        for r in resps:
            if isinstance(r, str):
                normalized.append([r])
            elif isinstance(r, (list, tuple)):
                # convert tuple -> list and ensure inner elements are strings
                normalized.append([str(x) for x in r])
            else:
                # fallback: coerce to string and wrap
                normalized.append([str(r)])
        resps = normalized

        # Optional debug help (uncomment when needed)
        # for i, inst in enumerate(resps):
        #     print(f"resps[{i}] type={type(inst)} len={len(inst)} sample={inst[0][:120]!r}")

        punct_tbl = dict.fromkeys(
            i for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)
            if self.ignore_case:
                st = st.lower()
            if self.ignore_punctuation:
                st = st.translate(punct_tbl)
            return st

        # use compiled regex from parent when available
        answer_regex = getattr(self, "regex", re.compile(self.regex_pattern))
        conf_regex = re.compile(self.confidence_pattern)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc.get("choices", [])
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)

            fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else re.compile("$^")
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({'|'.join(without_paren_fallback_regexes)})"
            ) if without_paren_fallback_regexes else re.compile("$^")

            # prefer a paired "LETTER, CONF" match
            paired_regex = re.compile(rf"(?<![A-Za-z])([A-Z])\s*,\s*{self.confidence_pattern}")

            filtered = []
            for resp in r:
                match = None
                conf = np.nan

                # 1) try paired "LETTER, CONF"
                pm = paired_regex.search(resp)
                if pm:
                    match = pm.group(1).strip()
                    # extract confidence from the paired substring
                    cm = conf_regex.search(pm.group(0))
                    if cm:
                        # confidence_pattern usually captures digits in group 1
                        try:
                            val = cm.group(1) if cm.groups() else cm.group(0)
                            conf = float(val)
                        except Exception:
                            conf = np.nan
                else:
                    # 2) fallback to robust letter extraction (same logic as MultiChoiceRegexFilter)
                    match = find_match(answer_regex, resp)
                    if not match:
                        match = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
                        if not match:
                            match = find_match(without_paren_fallback_regex, resp, without_paren_to_target)
                    if not match:
                        match = self.fallback

                    # try to find any confidence elsewhere in the response
                    conf_match = conf_regex.findall(resp)
                    if conf_match:
                        cm = conf_match[0]
                        if isinstance(cm, tuple):
                            cm = [c for c in cm if c][0]
                        try:
                            conf = float(cm)
                        except Exception:
                            conf = np.nan

                # clamp confidence
                if conf is not None and not np.isnan(conf):
                    conf = max(0.0, min(conf, 100.0))
                else:
                    conf = np.nan

                filtered.append((match, conf))

            filtered_resps.append(filtered)

        return filtered_resps
