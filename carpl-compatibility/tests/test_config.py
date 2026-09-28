import pytest
import yaml

from src.config import Parser

def test_parser_sets_string_attribute():
    conf = Parser({"model": "resnet50"})
    assert conf.model == "resnet50"

def test_parser_sets_numeric_attributes():
    conf = Parser({"lr": 0.001, "epochs": 10})
    assert conf.lr == 0.001
    assert conf.epochs == 10

def test_parser_preserves_list_value():
    conf = Parser({"bins": [0, 50, 70, 90]})
    assert conf.bins == [0, 50, 70, 90]

def test_parser_from_yaml_matches_dict_construction(tmp_path):
    data = {"lr": 0.001, "batch_size": 32, "img_size": 512}
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.dump(data))

    conf = Parser.from_yaml(str(yaml_file))

    assert conf.lr == 0.001
    assert conf.batch_size == 32
    assert conf.img_size == 512