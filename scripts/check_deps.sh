#!/usr/bin/env bash

set -e

# Use `pydeps` to make sure we're not importing any packages where we shouldn't be
pydeps --show-dot --no-show --no-config caikit | grep '\->' > deps.txt
trap "rm deps.txt" EXIT

if < deps.txt grep -q ".*caikit_runtime.*\->.*caikit_core.*"
then
    echo "Fail: The core is importing the runtime!"
    exit 1
fi

if < deps.txt grep -q ".*caikit_runtime.*\->.*caikit_interfaces.*"
then
    echo "Fail: The interfaces are importing the runtime!"
    exit 1
fi

if < deps.txt grep -q ".*caikit_interfaces.*\->.*caikit_core.module*"
then
    echo "Fail: The core module definitions are importing the interfaces!"
    exit 1
fi

if < deps.txt grep -q ".*caikit_interfaces.*\->.*caikit_core.data_model.*"
then
    echo "Fail: The core data model is importing the interfaces!"
    exit 1
fi

echo "Pass!"
