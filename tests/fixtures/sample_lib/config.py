# Standard
import dataclasses


@dataclasses.dataclass
class LittleConfig:
    library_name: str
    library_version: str


lib_config = LittleConfig(library_name="sample_lib", library_version="1.2.3")
