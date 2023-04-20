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


# TODO: Tests for:
# configure() without a file (get base)
# configure() with a file
# configure() again adds more
# configure() picks up env vars
# configure() with extra config files picks those up

# get_config() should be implicitly tested with those ^^

# Add `autouse` fixture that patch.object's `caikit.config.config._CONFIG` or something so we get a fresh one each time?
# - Need to make sure that these tests use a patch so that other concurrent tests are unaffected
