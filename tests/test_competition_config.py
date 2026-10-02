import pytest
from pydantic import ValidationError

from src.config import ServiceConfig, load_config


@pytest.fixture
def competition_config():
    config, _ = load_config("configs/service/jevsight-c.yaml", ServiceConfig)
    data = config.model_dump()
    data["competition_mode"] = True
    return data


@pytest.mark.parametrize("model", ["gpt-4o-mini", "openai/gpt-4o-mini"])
@pytest.mark.parametrize("embedding", ["openai/text-embedding-3-small", "another-embedding"])
def test_competition_accepts_mini_consolidation_and_unrestricted_embedding(competition_config, model, embedding):
    competition_config["llm"]["name"] = model
    competition_config["consolidation"]["decision_model"]["name"] = model
    competition_config["embedding"]["name"] = embedding

    config = ServiceConfig.model_validate(competition_config)

    assert config.competition_mode
    assert config.facts.temporal
    assert config.consolidation.mode == "llm"
    assert config.embedding.name == embedding


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o-mini-high", "typesafe/jev-1.13"])
def test_competition_rejects_other_extraction_models(competition_config, model):
    competition_config["llm"]["name"] = model

    with pytest.raises(ValidationError, match="competition_mode requires gpt-4o-mini"):
        ServiceConfig.model_validate(competition_config)


@pytest.mark.parametrize("model", ["gpt-4o", "typesafe/jev-1.13"])
def test_competition_rejects_other_consolidation_models(competition_config, model):
    competition_config["consolidation"]["decision_model"]["name"] = model

    with pytest.raises(ValidationError, match="competition_mode requires gpt-4o-mini"):
        ServiceConfig.model_validate(competition_config)


def test_competition_still_rejects_jev_mode(competition_config):
    competition_config["consolidation"]["mode"] = "jev"

    with pytest.raises(ValidationError, match="Jevsight is a local experiment"):
        ServiceConfig.model_validate(competition_config)


def test_local_jev_experiment_remains_valid():
    config, _ = load_config("configs/service/jevsight-d.yaml", ServiceConfig)

    assert not config.competition_mode
    assert config.consolidation.mode == "jev"
