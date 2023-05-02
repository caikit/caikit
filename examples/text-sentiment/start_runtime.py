from os import path
import sys
import alog

sys.path.append(path.abspath(path.join(path.dirname(__file__), "../"))) # Here we assume this file is at the same level of requirements.txt
import text_sentiment

alog.configure(default_level="debug")

from caikit.runtime import grpc_server
grpc_server.main()

