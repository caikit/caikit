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

# Standard
import argparse
import ast
import os
import re
import textwrap


class CustomDocstringConverter:
    """A class for converting custom style docstrings to Google style."""

    @staticmethod
    def extract_docstrings_from_file(file_path):
        """Extracts docstrings from the specified file.
        Args:
            file_path (str): Path to the file.
        Returns:
            list: List of extracted docstrings.
        """
        with open(file_path, "r") as file:
            file_contents = file.read()

        # Parse the file contents into an abstract syntax tree (AST)
        tree = ast.parse(file_contents)

        # Collect all docstrings
        docstrings = []
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
                and node.body
            ):
                # Check if the node has a docstring
                if isinstance(node.body[0], ast.Expr) and isinstance(
                    node.body[0].value, ast.Str
                ):
                    docstring = node.body[0].value.s
                    docstrings.append(docstring)

        return docstrings

    def convert_to_google_style(self, custom_docstring):
        """Converts a docstring to Google style.
        Args:
            custom_docstring (str): The original docstring.
        Returns:
            str: The converted docstring in Google style.
        """

        def ret_replacement(match):
            white_space, data_type, desc_white_space, description = match.groups()
            # If has already been converted, don't convert it again
            if ":" in data_type:
                return f"{white_space}{data_type}\n{desc_white_space}{description}"
            if description:
                # Clean up return and reformat it
                cleaned_description = re.sub(r"\s*\n\s*", " ", description.strip())
                wrapped_lines = textwrap.wrap(
                    f"{white_space}{data_type}: {cleaned_description}\n",
                    width=80,
                    subsequent_indent=desc_white_space,
                )
                returns_white_space = " " * (len(white_space) - 5)

                # Accounts for the fact Returns: may be last section
                if returns_last:
                    print("Returns last")
                    return "\n" + "\n".join(wrapped_lines) + "\n" + returns_white_space

                return "\n" + "\n".join(wrapped_lines)

            return f"{white_space}{data_type}\n"

        def arg_replacement(match):
            (
                white_space,
                name,
                arg_white_space,
                data_type,
                desc_white_space,
                description,
            ) = match.groups()
            if description:
                # Safety check if description is input incorrectly
                if (len(white_space.strip("\n"))) == len(desc_white_space):
                    return f"{white_space}{name} ({data_type})\n{desc_white_space}{description}"
                # Clean up argument and reformat it
                cleaned_description = re.sub(r"\s*\n\s*", " ", description.strip())
                wrapped_lines = textwrap.wrap(
                    f"{white_space}{name} ({data_type}): {cleaned_description}",
                    width=80,
                    subsequent_indent=desc_white_space,
                )
                # If there was a newline in white space, replace it
                if "\n" in white_space:
                    return "\n" + "\n".join(wrapped_lines)

                return "\n".join(wrapped_lines)

            return f"{white_space}{name} ({data_type})\n{desc_white_space}"

        converted_docstring = custom_docstring
        args = False
        returns = False
        notes = False
        examples = False
        returns_last = False

        # Determine args, returns, notes start
        if "Args:" in custom_docstring:
            args_start = custom_docstring.find("Args:")
            args = True
        if "Returns:" in custom_docstring:
            returns_start = custom_docstring.find("Returns:")
            returns = True
        if "Notes:" in custom_docstring:
            notes_start = custom_docstring.find("Notes:")
            notes = True
        if "Examples:" in custom_docstring:
            examples_start = custom_docstring.find("Examples:")
            examples= True

        # Find second """ (end of docstring)
        docstring_end = custom_docstring.find('"""', custom_docstring.find('"""') + 1)

        # Determine args ending
        if returns:
            args_end = returns_start
        elif notes:
            args_end = notes_start
        elif examples:
            args_end = examples_start
        else:
            args_end = docstring_end

        # Determine returns ending
        if notes:
            returns_end = notes_start
        elif examples:
            returns_end = examples_start
        else:
            returns_end = docstring_end
            returns_last = True

        # Extract the preceding white space, name, colon, data type, and following white
        # space of arguments using expressions
        arg_pattern = (
            r"(\s*)(\*\*\w+|\*\w+|\w+)(:\s*)(.*)\s*\n(\s*)((?!.*:)(?:.*(?:\n\5|$).*)*)?"
        )

        # Extract the data type and description of returns using regular expressions
        ret_pattern = r"(\s*)(.+)\n(\s+)((?:.*(?:\n(?!\n)|$).*)*)"

        if returns:
            returns_string = custom_docstring[returns_start + 8 : returns_end]

            # Replace the return pattern matches with the converted parts
            returns_string = re.sub(ret_pattern, ret_replacement, returns_string)
            converted_docstring = (
                custom_docstring[: returns_start + 8]
                + returns_string
                + custom_docstring[returns_end:]
            )

        if args:
            args_string = custom_docstring[args_start + 5 : args_end]

            # Replace args pattern matches with converted parts
            args_string = re.sub(
                arg_pattern, arg_replacement, args_string, flags=re.MULTILINE
            )
            converted_docstring = (
                converted_docstring[: args_start + 5]
                + args_string
                + converted_docstring[args_end:]
            )

        return converted_docstring

    def convert_docstrings(docstrings, file_path):
        """Converts docstrings to Google style and updates the file.
        Args:
            docstrings (list): List of docstrings to convert.
            file_path (str): Path to the file.
        """
        with open(file_path, "r") as file:
            file_contents = file.read()

        # Helper function to find the start and end position of a docstring
        def find_docstring_position(file_contents, docstring):
            start = file_contents.find(docstring)
            end = start + len(docstring)
            return start, end

        # Iterate over the docstrings and replace them in the file contents
        for docstring in docstrings:
            start, end = find_docstring_position(file_contents, docstring)
            converted_docstring = CustomDocstringConverter().convert_to_google_style(
                docstring
            )
            file_contents = (
                file_contents[:start] + converted_docstring + file_contents[end:]
            )

        # Update the file with the modified contents
        with open(file_path, "w") as file:
            file.write(file_contents)


def main():
    """The main function for running the Custom Docstring Converter."""

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Custom Docstring Converter")

    # Add an argument for the file path
    parser.add_argument("file_path", type=str, help="Path to the file to convert")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the file path or directory path from the arguments
    path = args.file_path

    # Check if the path is a file or directory
    if os.path.isfile(path):
        # Path is a file, convert only that file
        print("Updated: ", path, "\n")
        docstrings = CustomDocstringConverter.extract_docstrings_from_file(path)
        CustomDocstringConverter.convert_docstrings(docstrings, path)
    elif os.path.isdir(path):
        counter = 0
        # Path is a directory, convert all .py files in the directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    counter += 1
                    print("Updated: ", file_path, "\n")
                    docstrings = CustomDocstringConverter.extract_docstrings_from_file(
                        file_path
                    )
                    CustomDocstringConverter.convert_docstrings(docstrings, file_path)
        print("Done! Updated ", counter, " .py files")
    else:
        print("Invalid path provided.")


if __name__ == "__main__":
    main()
