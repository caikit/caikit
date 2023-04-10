# sys path hack here
# Standard
import os
import sys

protobufs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(protobufs_dir)
