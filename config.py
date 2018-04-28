# -*- coding:utf-8 -*-

import os
import configparser

current_dir = os.path.abspath(os.path.dirname(__file__))

class OperationalError(Exception):
    """operation error."""

class Dictionary(dict):
    """ custom dict."""

    def __getattr__(self, key):
        return self.get(key, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Config:
    def __init__(self, file_name="conf", cfg=None):
        """
        @param file_name: file name without extension.
        @param cfg: configuration file path.
        """
        #print(os.environ.items())
        env = {}
        for key, value in os.environ.items():
            if key.startswith("TEST_"):
                env[key] = value

        config = configparser.ConfigParser(env)

        if cfg:
            config.read(cfg)
        else:
            config.read(os.path.join(current_dir, "conf", "%s.ini" % file_name))

        for section in config.sections():
            setattr(self, section, Dictionary())
            for name, raw_value in config.items(section):
                try:
                    # Ugly fix to avoid '0' and '1' to be parsed as a
                    # boolean value.
                    # We raise an exception to goto fail^w parse it
                    # as integer.
                    if config.get(section, name) in ["0", "1"]:
                        raise ValueError

                    value = config.getboolean(section, name)
                except ValueError:
                    try:
                        value = config.getint(section, name)
                    except ValueError:
                        value = config.get(section, name)

                setattr(getattr(self, section), name, value)

    def get(self, section):
        """Get option.
        @param section: section to fetch.
        @return: option value.
        """
        try:
            return getattr(self, section)
        except AttributeError as e:
            raise OperationalError("Option %s is not found in "
                                   "configuration, error: %s" %
                                   (section, e))


if __name__ == "__main__":
    conf = Config()
    #print(conf.get("FILE_DATA").wav_path)

    #print(conf.get("FILE_DATA").label_file)

