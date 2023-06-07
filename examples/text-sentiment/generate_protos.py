import shutil
from caikit.runtime.dump_services import dump_services
import text_sentiment

shutil.rmtree("protos", ignore_errors=True)
dump_services("protos")