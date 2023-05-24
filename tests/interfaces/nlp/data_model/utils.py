# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
"""Shared utilities for data model testing.
"""


def validate_fields(obj):
    """Validate that the data model object, obj, has set all fields
    listed in obj.fields, which correspond to the protobuf fields.
    """
    return all((hasattr(obj, field) for field in obj.fields))
