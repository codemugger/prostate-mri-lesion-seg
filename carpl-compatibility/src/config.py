import yaml

class Parser:
    def __init__(self, config_dict):
        """
        Initialize from dictionary.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)