# ADR 024 Remote Module Invocation

Currently, caikit only supports locally loaded modules and does not understand a remote runtime 
server. Remote servers can allow for a better distribution of resources and more complex runtime 
architectures. Remote modules could also form the basis of a ["thin client"](https://github.com/caikit/caikit/issues/255) 
where a service can run remote invocations without installing library dependencies.
Library users have started creating custom initializers to solve these problems; however, 
those implementations depend on specific assumptions about the caikit runtime that might change 
from version to version.

## Decision

We propose implementing a system for remote model invocation directly in the Caikit core library. 
This system will handle discovering and describing models from a remote runtime. These remote modules 
will look and function like local modules, except any task or train invocation is forwarded to the 
remote. The core components of this system will be a new ModelFinder named RemoteModelFinder and a 
new ModelInitializer called RemoteModelInitializer.

The RemoteModelFinder will gather the remote server's connection and model information. In the first
iteration, the RemoteModelFinder will use the collected information to find a locally available 
ModuleBase and construct a new RemoteModuleConfig. By design, the RemoteModuleConfig does not 
contain any direct references to the imported Module and uses CaikitMethodSignatures to describe the
methods and tasks. This allows future implementations or other sources to construct RemoteModuleConfigs
without necessarily having to import the local ModuleBase. To adequately describe these methods, 
some assumptions are made about the Caikit service generation, specifically around the dataclass 
and rpc naming schemes.

The RemoteModuleConfig is then passed to the RemoteModelInitializer, which will construct a new 
RemoteModuleInstance with the same methods, signatures, and parameters as the source Module without 
using `caikit.runtime` or references to the original. This module will be constructed with the same 
`@module` and `@task.taskmethod` decorators to ensure the module acts precisely as a locally loaded 
module. One issue possible issue for the future is that the RemoteModelInstance relies on 
dataclasses created during service generation.



### RemoteModelFinder


```yaml
model_management:
    finders:
        <finder name>:
            type: REMOTE
            config:
                connection:
                    hostname: str <remote host>
                    port: Optional[int]=80/443 <remote port>
                    tls:
                        enabled: Optional[bool]=False <if ssl is enabled on the remote server>
                        ca_file: Optional[str]=None <path to remote ca file>
                        cert_file: Optional[str]=None <path to MTLS cert>
                        key_file: Optional[str]=None <path to MTLS key>
                        insecure_verify: Optional[bool] = False <if server's cert should be verified>
                    options:  Optional[Dict[str,str]]={} <optional dict of grpc or http configuration options>
                    timeout: Optional[int]=60 <optional client timeout setting>
                protocol: Optional[str]="grpc" <protocol the remote server is using (grpc or http)>
                model_key: Optional[str]=MODEL_MESH_MODEL_ID_KEY <Optional setting to override the grpc model name>
                min_poll_time: Optional[int]=30 <The minimum time between sending discovery requests>
                discover_models: Optional[bool]=True <bool to automatically discover remote models via the /info/models endpoint> 
                supported_models: Optional[Dict[str, str]]={} <mapping of model names to module_ids that this remote supports>
                    <model_path>: <module_id>
```

The proposed configuration for the RemoteModelFinder is above. The only required field is the 
generic `connection` dictionary that supports a secure channel, mutual TLS, and custom GRPC/HTTP 
options. The `connection.hostname` setting contains the remote's hostname, while `connection.port` determines the 
runtime port. The `connection.tls` dictionary contains all information 
related to TLS with `tls.enabled` controlling if the server is running SSL, `tls.ca_file` is the path to
the CA file that the remote's certificate is signed by, `tls.cert_file` is the path to the
MTLS client certificate to be sent with the request, and finally, `tls.key_file` which is the file 
containing the MTLS client key. The final connection config is `connection.options` which defines a 
list of options to pass to either the HTTP or GRPC request; for an example of options, take a look 
at the [GRPC Channel options](https://grpc.github.io/grpc/core/group__grpc__arg__keys.html#details)

There are three more optional parameters that help configure the remote connection. The first is an optional `protocol` config is used to select which protocol to send requests over, with the default being `grpc`. The next is `model_key` which is used to control the GRPC metadata field containing the model name, the default is ModelMeshs `mm-model-id`; however, a common alternative is `mm-vmodel-id`. The final parameter is the `min_poll_time` argument which controls how often to discover models. This stops the RemoteModelFinder from overloading the remote server. 

Two additional optional fields help control what models this remote supports. The 
`discover_models` setting is a boolean that controls if the finder should query the remote runtime
to dynamically discover what models are loaded and their corresponding `module_id`'s. The
`supported_models` config is a dictionary that contains a static mapping of model_paths to module_ids 
that the remote supports. The `supported_models`  setting is required to add support for remotes that
don't have a reflection api or ones that lazily load their models (like ModelMesh). 


To help illustrate the above config, we included some pseudo python code to illustrate what happens 
during model finding:
```python
def find_model(model_path: str)->RemoteModuleConfig:
    # Check if model_path is in static mapping
    if model_path in config.supported_models:
        local_module = module_registry().get(config.supported_models[model_path])
    
    # Check if  model can be discovered dynamically
    elif discover_models:
        remote_model_mapping = gather_remote_model_map(config.connection)
        local_module = remote_model_mapping.get(model_path)
    
    if not local_module:
        raise CaikitCoreException("Model not found")
    
    # Construct config for use by the RemoteModelInitializer. This function is 
    # described down below in the #RemoteModuleConfig section
    return generate_config_for_module(local_module, config.connection, model_path)
```

## RemoteModelInitializer

```yaml
model_management:
    initializers:
        <initializer name>:
            type: REMOTE
```

The proposed configuration for the RemoteModelInitializer is above. The initializer does not take in
any global configuration settings, as the remote information will be passed in via the 
RemoteModelFinder.  If a system is expected to have both local and remote models, consider using a 
MultiModelInitializer to handle both use cases.

To help illustrate how the RemoteModelInitializer would initialize a Module, we provided a snippet
of pseudo Python code:
```python3
def init(model_config: RemoteModuleConfig)->ModuleBase:
    # Construct empty RemoteModule Instance
    @module(
        id=model_config.id,
        name=model_config.name,
        version=model_config.name,
        task=[task_tuple[0] for task_tuple in model_config.task_methods]
    )
    class _RemoteModelInstance(RemoteModelBase):
        pass
    
    # Add all task methods to the RemoteModel class
    for task, inference_methods in model_config.task_methods:
        for method in inference_methods:
            infer_func = partial(_RemoteModelInstance.remote_method_request, method=method)
            task_wrapped_func = task.taskmethod(infer_func)
            setattr(_RemoteModelInstance, method.signature.name, task_wrapped_func)
    
    # Add train method to class if one exists
    if model_config.train_method:
        train_func = partial(_RemoteModelInstance.remote_method_request, method=model_config.train_method)
        setattr(_RemoteModelInstance, model_config.train_method.signature.name, train_func)
    
    # Return Model Instance
    return _RemoteModelInstance(model_config.connection, model_config.model_path)    
    

class RemoteModelBase(ModuleBase):
   def __init__(self, connection: Dict[str, Any], model_path: str):
        ...
   
   def remote_method_request(self, method: RemoteMethodRpc, *args, **kwargs):
        # Run the remote invocation using the information defined in the RemoteMethodRpc
        if self.connection.protocol == "grpc":
            <run remote `method` using grpc>
        elif self.connection.protocol == "http":
            <run remote `method` using http>
```

## RemoteModuleConfig

```yaml
class RemoteModuleConfig(ModuleConfig):
  # Remote runtime information copied from the RemoteModelFinder config
  connection: Dict[str, Any]

  # Method information
  # use list and tuples instead of a dictionary to avoid aconfig.Config error
  task_methods: List[Tuple[type[TaskBase], List[RemoteMethodRpc]]]
  train_method: RemoteMethodRpc
  
  # Source Module Information
  module_id: str
  module_name: str
  model_path: str

@dataclass
class RemoteMethodRpc:
    # full signature for this RPC
    signature: CaikitMethodSignature
    
    # Request and response objects for this RPC
    request_dm_name: str
    response_dm_name: str
    
    # Either the function name of the GRPC Servicer or HTTP endpoint
    rpc_name: str

    # Only used for infer RPC types
    input_streaming: bool
    output_streaming: bool
```

The RemoteModuleConfig is a custom subclass of ModuleConfig, which contains a description of a Module's
tasks, inference and train methods, and version information, as well as the connection information of 
the remote runtime. The combination of the two allows the RemoteModelInitializer to construct a new 
RemoteModule without having to import `caikit.runtime` or the source model. To help simplify the
config definition and access, we decided to create a helper dataclass, `RemoteMethodRpc`, which contains
information about a specific method and includes things like the CaikitMethodSignature, 
request&response DataModel names, and the remote RPC name. The `RemoteMethodRpc` dataclass contains 
all the runtime-specific assumptions  and is the main point where overlap happens. The 
`RemoteModelFinder`, which constructs `RemoteMethodRpc`s, should not import `caikit.runtime`; 
however, it will re-use a lot of code around the different naming schemes.

We created the following pseudocode to help illustrate how the `RemoteModelFinder` constructs a 
`RemoteModuleRpc`. (Note this is the function used in the pseudocode for the [RemoteModelFinder](#remotemodelfinder))

```python
def generate_config_for_module(module: ModuleBase, connection_info: Dict[str, Any], model_path: str)->RemoteModuleConfig:
    # Gather a description of all tasks and their associated methods
    task_methods = []
    for task_class in module.tasks:
        task_functions = []
        for input, output, signature in module_class.get_inference_signatures(task_class):
            # Construct request_dm_name,  task_request_name, and rpc_name. This makes assumptions
            # about caikit.runtime service generation
            request_dm_name ~= "TaskRequest"
            task_request_name ~= "TaskPredict"
            rpc_name = "/api/v1/TaskPredict"
            
            task_functions.append(
                RemoteMethodRpc(
                    signature=signature,
                    request_dm_name=request_dm_name,
                    response_dm_name=signature.return_type.__name__,
                    rpc_name=rpc_name,
                    input_streaming=input,
                    output_streaming=output,
                )
            )
            
        task_functions.append((task_class, task_functions))
    
    # Gather description of the Train functions
    train_method = None
    if module.TRAIN_SIGNATURE:
        request_dm_name ~= "TrainRequest"
        rpc_name ~= "/api/v1/TaskTrain"
        
        train_method = RemoteMethodRpc(
            signature=module.TRAIN_SIGNATURE,
            request_dm_name=request_dm_name,
            response_dm_name=module.TRAIN_SIGNATURE.return_type.__name__,
            rpc_name=rpc_name,
        )
    
    # Construct the remote config
    return RemoteModuleConfig(
        {
            # Connection info
            "connection":connection_info,
            # Method info
            "task_methods":task_methods,
            "train_method":train_method,
            # Source Module Information
            "model_path": model_path,
            "module_id": module.MODULE_ID,
            "module_name": module.MODULE_NAME,
        }
    )
        
```

### Diagram

<img width="1439" alt="image" src="https://github.com/caikit/caikit/assets/37811263/4e0d0573-04fa-42fa-bdce-fcd989b9bbd6">
This is an updated block diagram of the various model loading components and their relationships. 


## Status

choose one: Accepted

if deprecated, include a rationale.

If superseded, include a link to the new ADR


## Consequences

- Library users will be able to configure remote runtime servers.
- Multiple caikit runtimes can work together to serve a large set of models.
- When updating or changing the service generation, the `RemoteModelFinder` will also have to be changed.
