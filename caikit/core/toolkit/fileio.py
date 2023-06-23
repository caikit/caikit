# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Basic routines for reading and writing common file types.
"""

# Standard
import csv
import json
import os
import pickle
import shutil

# Third Party
import yaml


def load_txt(filename):
    """Load a string from a file with utf8 encoding."""
    with open(filename, mode="r", encoding="utf8") as fh:
        return fh.read()


def load_txt_lines(filename):
    """Load a list of files from a text file with utf8 encoding"""
    with open(filename, mode="r", encoding="utf8") as fh:
        wordlist = list(map(str.strip, fh.readlines()))
    return wordlist


def save_txt(text, filename, mode="w"):
    """Write a string to a text file with utf8 encoding."""
    with open(filename, mode=mode, encoding="utf8") as fh:
        fh.write(text)


def load_binary(filename):
    """Load a binary string from a file."""
    with open(filename, mode="rb", encoding="utf-8") as fh:
        return fh.read()


def save_binary(data, filename):
    """Write a binary buffer to a file."""
    with open(filename, mode="wb", encoding="utf-8") as fh:
        fh.write(data)


def load_csv(filename):
    """Load a csv into a list-of-lists."""
    with open(filename, mode="r", newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter=",", quotechar='"'))


def save_csv(text_list, filename, mode="w"):
    """Write a list-of-lists to a csv file."""
    with open(filename, mode=mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=",")
        writer.writerows(text_list)


def load_dict_csv(filename):
    """Load a csv into a list-of-dicts."""
    with open(filename, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return list(csv_reader)


def save_dict_csv(dict_list, filename, mode="w"):
    """Write a list of dicts to a csv file."""
    if dict_list:
        keys = dict_list[0].keys()
        with open(filename, mode=mode, encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dict_list)


def load_json(filename):
    """Load a json file into a dictionary."""
    with open(filename, mode="r", encoding="utf8") as fh:
        return json.load(fh)


def save_json(save_dict, filename, mode="w"):
    """Save a dictionary into a json file."""
    with open(filename, mode=mode, encoding="utf8") as fh:
        json.dump(save_dict, fh, indent=2, ensure_ascii=False)


def load_yaml(filename):
    """Load a yaml file into a dictionary."""
    with open(filename, mode="r", encoding="utf8") as fh:
        return yaml.safe_load(fh)


def save_yaml(save_dict, filename, mode="w"):
    """Save a dictionary into a yaml file."""
    with open(filename, mode=mode, encoding="utf8") as fh:
        yaml.safe_dump(save_dict, fh, default_flow_style=False)


def load_pickle(filename):
    """Load an object from a pickle file."""
    with open(filename, mode="rb") as fh:
        return pickle.load(fh)


def save_pickle(obj, filename, mode="wb"):
    """Save an object to a pickle file."""
    # pylint: disable=unspecified-encoding
    with open(filename, mode=mode) as fh:
        pickle.dump(obj, fh)


def save_raw(save_content, filename, mode="w"):
    """Write the given raw string content to output file."""
    with open(filename, mode=mode, encoding="utf8") as fh:
        fh.write(save_content)


def compress(dir_path, output_path=None, extension="zip"):
    """Compress a given folder recursively to an archive with a given extension format

    Args:
        dir_path (str): Path of directory to compress
        output_path: (Optional) str
            Output path where the archive is created. Defaults to current path + 'archive' +
            format extension
            >>> compress('.', 'my/path', 'tar')
            >>> # saves to 'my/path/archive.tar'

        extension: (Optional) (one of: zip/tar/gztar/bztar/xztar depending on module availability)
            Defaults to .zip

    Returns:
        str: Path to created archive
    """
    if not output_path:
        output_path = os.path.join(os.getcwd(), "archive")

    # Strip away anything preceding '.'
    extension = extension.split(".")[-1]

    shutil.make_archive(output_path, extension, dir_path)
    return output_path + "." + extension
