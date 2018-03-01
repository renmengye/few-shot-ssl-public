from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

CONFIG_REGISTRY = {}


def RegisterConfig(dataset_name, model_name):
  """Registers a configuration."""

  def decorator(f):
    key = "{}_{}".format(dataset_name, model_name)
    CONFIG_REGISTRY[key] = f
    return f

  return decorator


def get_config(dataset_name, model_name):
  key = "{}_{}".format(dataset_name, model_name)
  if key in CONFIG_REGISTRY:
    return CONFIG_REGISTRY[key]()
  else:
    raise ValueError("No registered config: \"{}\"".format(key))
