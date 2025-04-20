import yaml

def load_config():
    """
    load project config from yaml file
    """
    YAML_CONFIG_PATH = "./config.yaml"
    with open(YAML_CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config