import pandas as pd
import pytest

from geotool import companion_platforms


class FakeGSM:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeGSE:
    def __init__(self, gsms):
        self.gsms = gsms


def _gsm(platform_id: str, title: str) -> FakeGSM:
    return FakeGSM({"platform_id": [platform_id], "title": [title]})


class TestCompanionPlatformsPresent:
    def test_pair_present(self):
        assert companion_platforms.companion_platforms_present(["GPL96", "GPL97", "GPL570"]) == [
            ("GPL96", "GPL97")
        ]

    def test_only_one_half_present_is_not_a_pair(self):
        assert companion_platforms.companion_platforms_present(["GPL96", "GPL570"]) == []

    def test_unrelated_platforms(self):
        assert companion_platforms.companion_platforms_present(["GPL570", "GPL571"]) == []


class TestMatchCompanionSamples:
    def test_matches_suffix_convention(self):
        # GSE43288-style: "N1_A" / "N1_B"
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "N1_A"),
            "GSMA2": _gsm("GPL96", "PaCa1_A"),
            "GSMB1": _gsm("GPL97", "N1_B"),
            "GSMB2": _gsm("GPL97", "PaCa1_B"),
        })
        pairing = companion_platforms.match_companion_samples(gse, "GPL96", "GPL97")
        assert pairing == {"GSMA1": "GSMB1", "GSMA2": "GSMB2"}

    def test_matches_embedded_token_convention(self):
        # GSE1124-style: "controls NA01 U133A array" / "controls NA01 U133B array"
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "controls NA01 U133A array"),
            "GSMB1": _gsm("GPL97", "controls NA01 U133B array"),
        })
        pairing = companion_platforms.match_companion_samples(gse, "GPL96", "GPL97")
        assert pairing == {"GSMA1": "GSMB1"}

    def test_unequal_sample_counts_refuses_to_match(self):
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "N1_A"),
            "GSMA2": _gsm("GPL96", "N2_A"),
            "GSMB1": _gsm("GPL97", "N1_B"),
        })
        assert companion_platforms.match_companion_samples(gse, "GPL96", "GPL97") is None

    def test_no_common_title_key_refuses_to_match(self):
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "sample one"),
            "GSMB1": _gsm("GPL97", "sample two"),
        })
        assert companion_platforms.match_companion_samples(gse, "GPL96", "GPL97") is None

    def test_missing_platform_refuses_to_match(self):
        gse = FakeGSE({"GSMA1": _gsm("GPL96", "N1_A")})
        assert companion_platforms.match_companion_samples(gse, "GPL96", "GPL97") is None

    def test_duplicate_normalized_titles_refuse_to_match(self):
        # two GPL96 samples that normalize to the same key -- ambiguous, don't guess
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "Sample_A"),
            "GSMA2": _gsm("GPL96", "Sample_A"),
            "GSMB1": _gsm("GPL97", "Sample_B"),
            "GSMB2": _gsm("GPL97", "Sample_B"),
        })
        assert companion_platforms.match_companion_samples(gse, "GPL96", "GPL97") is None


class TestDetectPairings:
    def test_detects_and_matches_present_pair(self):
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "N1_A"),
            "GSMB1": _gsm("GPL97", "N1_B"),
        })
        result = companion_platforms.detect_pairings(gse)
        assert result == {("GPL96", "GPL97"): {"GSMA1": "GSMB1"}}

    def test_no_pair_present(self):
        gse = FakeGSE({"GSM1": _gsm("GPL570", "sample 1")})
        assert companion_platforms.detect_pairings(gse) == {}

    def test_pair_present_but_unmatchable_is_omitted(self):
        gse = FakeGSE({
            "GSMA1": _gsm("GPL96", "N1_A"),
            "GSMA2": _gsm("GPL96", "N2_A"),
            "GSMB1": _gsm("GPL97", "N1_B"),
        })
        assert companion_platforms.detect_pairings(gse) == {}


class TestCombinePairedProbeColumns:
    def test_unions_disjoint_probe_sets(self):
        probe_matrix = pd.DataFrame(
            {
                "GSMA1": [1.0, 2.0, None],
                "GSMB1": [None, None, 3.0],
            },
            index=["probe_a1", "probe_a2", "probe_b1"],
        )
        combined = companion_platforms.combine_paired_probe_columns(
            probe_matrix, {"GSMA1": "GSMB1"}
        )
        assert list(combined.columns) == ["GSMA1+GSMB1"]
        assert combined.loc["probe_a1", "GSMA1+GSMB1"] == 1.0
        assert combined.loc["probe_a2", "GSMA1+GSMB1"] == 2.0
        assert combined.loc["probe_b1", "GSMA1+GSMB1"] == 3.0

    def test_overlapping_probe_keeps_first_platforms_value(self):
        probe_matrix = pd.DataFrame({"GSMA1": [1.0], "GSMB1": [9.0]}, index=["shared_probe"])
        combined = companion_platforms.combine_paired_probe_columns(
            probe_matrix, {"GSMA1": "GSMB1"}
        )
        assert combined.loc["shared_probe", "GSMA1+GSMB1"] == 1.0

    def test_unpaired_columns_pass_through(self):
        probe_matrix = pd.DataFrame(
            {"GSMA1": [1.0], "GSMB1": [2.0], "GSMC1": [3.0]}, index=["p1"]
        )
        combined = companion_platforms.combine_paired_probe_columns(
            probe_matrix, {"GSMA1": "GSMB1"}
        )
        assert set(combined.columns) == {"GSMA1+GSMB1", "GSMC1"}
        assert combined.loc["p1", "GSMC1"] == 3.0

    def test_no_pairing_returns_matrix_unchanged(self):
        probe_matrix = pd.DataFrame({"GSM1": [1.0]}, index=["p1"])
        combined = companion_platforms.combine_paired_probe_columns(probe_matrix, {})
        assert combined is probe_matrix


class TestCollapsePairedSamples:
    def test_collapses_pair_into_one_row(self):
        samples = pd.DataFrame(
            [
                {"gsm_id": "GSMA1", "platform_id": "GPL96", "title": "N1_A"},
                {"gsm_id": "GSMB1", "platform_id": "GPL97", "title": "N1_B"},
                {"gsm_id": "GSMC1", "platform_id": "GPL570", "title": "other"},
            ]
        )
        pairings = {("GPL96", "GPL97"): {"GSMA1": "GSMB1"}}
        collapsed = companion_platforms.collapse_paired_samples(samples, pairings)
        assert len(collapsed) == 2
        merged_row = collapsed[collapsed["gsm_id"] == "GSMA1+GSMB1"].iloc[0]
        assert merged_row["platform_id"] == "GPL96+GPL97"
        assert merged_row["title"] == "N1_A"  # kept from the "a" row
        assert "GSMC1" in collapsed["gsm_id"].tolist()

    def test_no_pairings_returns_samples_unchanged(self):
        samples = pd.DataFrame([{"gsm_id": "GSM1", "platform_id": "GPL570"}])
        result = companion_platforms.collapse_paired_samples(samples, {})
        assert result is samples
