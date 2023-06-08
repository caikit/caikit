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

        # # Extract the name, data type, and description of arguments using regular expressions
        # arg_pattern = r"(?!Args)(?!rgs)(?!gs)(?!s)(\*\*\w+|\*\w+|\w+):\s*((?:\w+\s*\
        #     |\s*)*(?:\w+\s*\|)?\s*(?:\w+(?:\.\w+)*\s+\([^)]+\)|\w+(?:\.\w+)*|\w+(?:\
        #         .\w+)*\s*\|\s*\w+(?:\.\w+)*))\s*\n\s*((?:(?!:\s*Returns:\s).)*?)($|\n)"

        # # Extract the white space, data type, and description of returns using regular expressions
        # ret_pattern = r"Returns:(\s*)(?!.*:)([^:\n]+)\n(\s+)((?:.*(?:\n(?!\n)|$).*)*)(\n|\"\"\")"

        # def arg_replacement(match):
        #     name, data_type, description, _ = match.groups()
        #     return f"{name} ({data_type}): {textwrap.dedent(description)}\n"

        # # Replace the argument pattern matches with the converted parts
        # converted_docstring = re.sub(arg_pattern, arg_replacement, converted_docstring)

        # return converted_docstring

        # converted_docstring = custom_docstring

        # # If "Args:" is in custom_docstring
        #     # Want to grab arguments name (first), argument data type (second, comes after colon), argument description (on new line)
        #     # Based on indentation, know if it's description or argument name: argument type
        #         # 1 more indent than Args: arguments name: argument type
        #         # 2 more indents than Args: argument description
        #     # Gather arguments as one strings and format each one individually using textwrap
        #     # We know Args are done when we see the words "Returns:", "Notes:", or """
        
        # # If "Returns:" is in custom_docstring
        #     # Grab returns data type, returns description
        #     # Based on indentation, know if it's description or return data type
        #         # 1 more indents than Returns: returns data type
        #         # 2 more indents than Returns: returns description
        #     # Gather returns as one string and format using textwrap
        #     # Know that Returns is done when we see words "Notes:" or """
        
        # docstring_list = custom_docstring.split('\n')

        def ret_replacement(match):
            white_space, data_type, desc_white_space, description = match.groups()
            if description:
                cleaned_description = re.sub(r"\s*\n\s*", " ", description.strip())
                wrapped_lines = textwrap.wrap(
                    f"{white_space}{data_type}: {cleaned_description}\n", width=72
                )
                indented_lines = textwrap.indent(
                    "\n".join(wrapped_lines[1:]), " " * (len(desc_white_space))
                )
                returns_white_space = (" " * (len(white_space)-5))
                # Accounts for the fact Returns: may be last section
                if(returns_last):
                    return f"Returns:\n{wrapped_lines[0]}\n{indented_lines}\n{returns_white_space}"
                else:
                    return f"Returns:\n{wrapped_lines[0]}\n{indented_lines}"
            else:
                return f"Returns:{white_space}{data_type}\n"

        converted_docstring=custom_docstring
        args= False
        returns = False
        notes = False
        returns_last = False

        # Determine args, returns, notes start
        if("Args:" in custom_docstring):
            args_start = custom_docstring.find("Args:")
            args = True
            print("Args start: ",args_start)
        if("Returns:" in custom_docstring):
            returns_start = custom_docstring.find("Returns:")
            returns = True
            print("Returns start: ",returns_start)
        if("Notes:" in custom_docstring):
            notes_start = custom_docstring.find("Notes:")
            notes = True
            print("Notes start: ", notes_start)

        docstring_end = custom_docstring.find('"""',custom_docstring.find('"""')+1)

        # Determine args ending
        if(returns):
            args_end = returns_start
        elif(notes):
            args_end = notes_start
        else:
            args_end = docstring_end

        #Determine returns ending
        if(notes):
            returns_end = notes_start
        else:
            returns_end = docstring_end
            returns_last = True

        # # Extract the name and data type of arguments using regular expressions
        # arg_pattern = r"(?!Args)(?!rgs)(?!gs)(?!s)(\*\*\w+|\*\w+|\w+):\s*((?:\w+\s*\
        #     |\s*)*(?:\w+\s*\|)?\s*(?:\w+(?:\.\w+)*\s+\([^)]+\)|\w+(?:\.\w+)*|\w+(?:\
        #         .\w+)*\s*\|\s*\w+(?:\.\w+)*))\s*\n\s*($|\n)"
        
        # Extract the data type and description of returns using regular expressions
        ret_pattern = r"(\s*)(?!.*:)([^:\n]+)\n(\s+)((?:.*(?:\n(?!\n)|$).*)*)"

        returns_string = custom_docstring[returns_start+8:returns_end]
        print(returns_string)

        # Replace the return pattern matches with the converted parts
        returns_string = re.sub(ret_pattern, ret_replacement, returns_string)
            
        converted_docstring = (
            custom_docstring[:returns_start] + returns_string + custom_docstring[returns_end:]
        )

        print("Docstring end: ", docstring_end)
        # docstring_end-1 will be a blank character!
        print("Docstring end character: ", custom_docstring[docstring_end])
        print("Args end: ", args_end)
        print("Returns end: ", returns_end)
        

        

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
