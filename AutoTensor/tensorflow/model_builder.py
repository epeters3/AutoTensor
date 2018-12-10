import tensorflow as tf
from tensorflow import keras

from AutoTensor.q_learning.config_scheme import SchemeManager

scheme_manager = SchemeManager()


def instantiate_config(config, class_map):
    """
    Makes an exact copy of config, but every time a class is referenced,
    instantiates the actual class. Returns a settings dictionary that
    can be used by model_builder.
    """
    if isinstance(config, dict):
        if "class_name" in config.keys():

            # This is a dict representing a class. Instantiate
            # that class using the class references found in class_map
            return class_map[config["class_name"]](
                **instantiate_config(config["args"], class_map))

        else:

            settings = {}
            for key, value in config.iteritems():
                if isinstance(value, dict):
                    settings[key] = instantiate_config(value, class_map)
                elif isinstance(value, list):
                    settings[key] = []
                    for val in value:
                        if isinstance(val, dict):
                            settings[key].append(
                                instantiate_config(val, class_map))
                        else:
                            settings[key].append(val)
                else:
                    settings[key] = value
            return settings


def model_builder(config, num_classes):
    class_map = scheme_manager.get_class_map()
    settings = instantiate_config(config, class_map)

    settings["compile_args"]["metrics"] = ["accuracy"]
    model = keras.Sequential(
        settings["layers"] +
        [keras.layers.Dense(num_classes, activation="softmax")])
    model.compile(**settings["compile_args"])
    return model


# model = keras.Sequential(
# [
#     keras.layers.Conv2D(64, kernel_size=5, input_shape=(65,65,3), activation="relu"),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(64, kernel_size=5, activation="relu"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Flatten()
# ] + settings["layers"] +
# [keras.layers.Dense(num_classes, activation="softmax")])