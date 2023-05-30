# ADR 001: Module

## Overview

A module wraps implementation for a given AI model algorithm 'type' for the Caikit runtime. For instance, syntax (tokenization, parts of speech tagging, dependency parsing, lemmatization), entity extraction,
keywords extraction, relations, sentiments and so on. Each 'type' is considered an isolated piece which can be used in conjunction with other module 'types' as part of a Caikit operation.

It should be noted that a module 'type', for instance entity mention extraction, can have multiple implementations, which we refer to as module 'classes', and each such implementation gets its own:
`module_id` This is an UUID which is registered for each module class and used by the runtime to lookup the class from a model config, inconjunction with `module_name` and `module_version`.

```python3
import caikit

@caikit.module(
    "00110203-0405-0607-0809-0a0b02dd0e0f", "MyModule", "0.0.1")
class MyModule(caikit.core.ModuleBase):
    # my impl

```

A module wraps one or more AI algorithms, and provides a crisply defined interface for obtaining predictions by providing one or more data model objects as input and returns exactly one data model object as output.  

## Interface

Currently, each module implements the following APIs, some of which can be considered core (implemented by all moduyles),
such as `.__init__()`, `.run()` and some which may not be present in all (like `.train()`). Additionally, modules come with some helper methods that are useful in analyzing block performance.

#### `__init__()`

`__init__` is the constructor for the module and should take only in-memory data structures, relevant to the block implementation.  
Since python does not have constructor overloading, having a primitive, in-memory constructor allows us to implement multiple ways of instantiating blocks by using classmethods.

#### `load()`

`load` is a `classmethod` that loads the AI model by reading the `config.yml` and any artifacts into memory and then calls the module class' `__init__` method to create and return a new loaded instance of the model. The structure of `config.yml` is standardized to some extent, with some other pieces which can be specified as per class requirements.

#### `run()`:
Takes instances of classes defined as [data models](https://github.com/caikit/caikit/blob/main/docs/adrs/010-data-model-definition.md) as inputs and returns a single instance of [data model](https://github.com/caikit/caikit/blob/main/docs/adrs/010-data-model-definition.md) as output.

Different algorithms of the same module type return the same type of output, but may take different kinds of inputs.
This may feel a little overly prescriptive, but it allows us to keep things organized and also to have a generic server that can swap out different algorithms for a given module type.  
So, `run` is very much like a `predict` method in other frameworks and we often use them in our verbiage interchangeably.

#### `train()`

`train` is a python `@classmethod` that takes one or more [data stream]() instances as inputs containing training data and, potentially, other necessary arguments.
The `train` method creates a new instance of the module, in memory, that can later be saved to disk using `save()`.  
So, `train` is similar to `fit` in some frameworks. Not all modules will have a `train` implemented.

#### `save()`

This API is used to serialize the loaded in-memory model to disk by saving a `config.yml` file along with any necessary artifacts to a provided directory path.  
We have a `caikit.core.modules.ModuleSaver` class that provides most of this functionality in a consistent and easy way, along with helper methods to copy files/objects. Inner algorithms provide ways to serialize an in-memory model to disk, and those 'serializers' can be used by implementing `caikit.core.toolkit.ObjectSerializer`.

```python3

from caikit.core import ModuleSaver

class MyModule:
    
    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            rel_path, _ = module_saver.add_dir("model_name")
            save_path = os.path.join(model_path, rel_path)
            self.sentiment_pipeline.save_pretrained(save_path)
            module_saver.update_config({"artifact_path": rel_path})

```

#### `evaluate_quality()`

This is a helper API, implemented at a module base level (`caikit.core.ModuleBase`) which allows module level
evaluation of a model's quality (return macro/micro precision/recall/F1 scores). However, it requires a module 'type' level implementation of certain helper methods and initialization of some types to achieve it. The `caikit.toolkit.QualityEvaluator` base class specifies methods and objects needed for the module's quality evaluation, including:

1. `evaluation_type` and `evaluator`: Used to specify type of evaluation

```python
evaluation_type = caikit.toolkit.EvalTypes.MULTILABEL_MULTICLASS # (or SINGLECLASS)
evaluator = caikit.toolkit.QualityEvaluator # (or specify a custom evaluator)
```

2. `_load_evaluation_dataset(dataset_path)`: Load the test dataset from a given path and provide as a datastream object

3. `find_label_func(*args)`: A `@staticmethod` to extract the 'label' from an entry of test dataset stream

4. `find_label_data_func(label, *args)`: `@staticmethod` to extract data belonging to a particular 'label'

5. `_extract_gold_set(dataset)`: `@classmethod` to extract ground truth (gold standard data) from dataset stream

6. `_extract_pred_set(dataset, pre_process_function)`: Instance method to get predicted results from the dataset's input, after running them against the loaded module.

One can call a module's `.evaluate_quality(dataset_path, preprocess_func)` specifying the test dataset path, and internally the module type's base class methods will be triggered to obtain the gold labels and predicted labels, and perform evaluation to return quality metrics.


#### `stream()`

This is a helper method which works in conjunction with `caikit.core.data_model.streams.DataStream` to allow running large sets of data for a given model using streams. Refer to [data stream]() for details.

## Status

Approved

## Consequences

- Caikit will support modules as first-class constructs
- Modules contain atomic operations
