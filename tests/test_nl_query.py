from geotool import nl_query


class _FakeTextBlock:
    def __init__(self, parsed_output):
        self.type = "text"
        self.parsed_output = parsed_output


class _FakeResponse:
    def __init__(self, parsed_output):
        self.content = [_FakeTextBlock(parsed_output)]


class _FakeMessages:
    def __init__(self, results):
        # a list lets classify_series_with_escalation's two calls return different things
        self._results = list(results) if isinstance(results, list) else [results]
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results[min(len(self.calls) - 1, len(self._results) - 1)]
        return _FakeResponse(result)


class _FakeClient:
    def __init__(self, results):
        self.messages = _FakeMessages(results)


def make_classification(**overrides):
    defaults = dict(
        matches_diagnosis=True,
        diagnosis_detail="pancreatic ductal adenocarcinoma",
        species="human",
        sample_type="biopsy",
        tissue_class="tissue",
        assay_type="bulk_rnaseq",
        selection_method="none",
    )
    defaults.update(overrides)
    return nl_query.SeriesClassification(**defaults)


def test_parse_query_filters_returns_filters_from_mocked_call(monkeypatch):
    filters = nl_query.QueryFilters(
        diagnosis="pancreatic cancer",
        diagnosis_synonyms=["pancreatic ductal adenocarcinoma", "PDAC"],
        species="human",
        sample_type="biopsy",
        min_samples=20,
    )
    fake_client = _FakeClient(filters)
    monkeypatch.setattr(nl_query.anthropic, "Anthropic", lambda: fake_client)

    result = nl_query.parse_query_filters("Human biopsy pancreatic cancer cohorts with sample size more than 20")

    assert result == filters
    assert len(fake_client.messages.calls) == 1


def test_build_summary_prompt_includes_key_fields():
    candidate = {"gse_id": "GSE1", "title": "A study", "organism": "Homo sapiens", "summary": "Some summary", "n_samples": 42}
    prompt = nl_query.build_summary_prompt(candidate, ["Illumina NovaSeq X Plus"])
    assert "GSE1" in prompt
    assert "A study" in prompt
    assert "Some summary" in prompt
    assert "42" in prompt
    assert "Illumina NovaSeq X Plus" in prompt


def test_classify_series_returns_parsed_classification(monkeypatch):
    classification = make_classification()
    fake_client = _FakeClient(classification)
    monkeypatch.setattr(nl_query.anthropic, "Anthropic", lambda: fake_client)

    filters = nl_query.QueryFilters(diagnosis="pancreatic cancer", diagnosis_synonyms=["PDAC"])
    candidate = {"gse_id": "GSE1", "title": "PDAC study", "summary": "...", "n_samples": 30}

    result = nl_query.classify_series(candidate, filters)
    assert result == classification
    assert len(fake_client.messages.calls) == 1
    # diagnosis + synonyms should be baked into the system prompt sent to Claude
    assert "pancreatic cancer" in fake_client.messages.calls[0]["system"]
    assert "PDAC" in fake_client.messages.calls[0]["system"]


def test_classify_series_with_escalation_skips_second_call_when_confident(monkeypatch):
    confident = make_classification()
    fake_client = _FakeClient(confident)
    monkeypatch.setattr(nl_query.anthropic, "Anthropic", lambda: fake_client)

    filters = nl_query.QueryFilters(diagnosis="pancreatic cancer")
    candidate = {"gse_id": "GSE1", "title": "t", "summary": "s"}

    result = nl_query.classify_series_with_escalation(candidate, filters, escalate=True)
    assert result == confident
    assert len(fake_client.messages.calls) == 1  # no escalation needed


def test_classify_series_with_escalation_reruns_on_escalation_model_when_ambiguous(monkeypatch):
    ambiguous = make_classification(species="unknown", sample_type="unknown", tissue_class="unknown")
    confident = make_classification()
    fake_client = _FakeClient([ambiguous, confident])
    monkeypatch.setattr(nl_query.anthropic, "Anthropic", lambda: fake_client)

    filters = nl_query.QueryFilters(diagnosis="pancreatic cancer")
    candidate = {"gse_id": "GSE1", "title": "t", "summary": "s"}

    result = nl_query.classify_series_with_escalation(candidate, filters, escalate=True)
    assert result == confident
    assert len(fake_client.messages.calls) == 2
    assert fake_client.messages.calls[1]["model"] == nl_query.config.LLM_ESCALATION_MODEL


def test_classify_series_with_escalation_does_not_escalate_when_flag_off(monkeypatch):
    ambiguous = make_classification(species="unknown", sample_type="unknown", tissue_class="unknown")
    fake_client = _FakeClient(ambiguous)
    monkeypatch.setattr(nl_query.anthropic, "Anthropic", lambda: fake_client)

    filters = nl_query.QueryFilters(diagnosis="pancreatic cancer")
    candidate = {"gse_id": "GSE1", "title": "t", "summary": "s"}

    result = nl_query.classify_series_with_escalation(candidate, filters, escalate=False)
    assert result == ambiguous
    assert len(fake_client.messages.calls) == 1
