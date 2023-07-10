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
from typing import Callable, Optional
import traceback

# Third Party
from grpc._utilities import RpcMethodHandler
from prometheus_client import Gauge
import grpc

# First Party
import alog

# Local
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import CaikitRPCBase
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("SERVER-WRAPR")

IN_PROGRESS_GAUGE = Gauge(
    "rpc_in_progress_gauge",
    "Total number of in-flight requests to caikit-runtime",
    ["rpc_name"],
)


class CaikitRuntimeServerWrapper(grpc.Server):
    """This class wraps an underlying gRPC server for the purpose of
    intercepting the binding of servicers (e.g., the CaikitRuntimeServicer) to
    the server so that the RPC handlers that are registered to the server
    can optionally be replaced with a generic global predict RPC handler
    instead.
    """

    def __init__(self, server, global_predict, intercepted_svc_package: ServicePackage):
        """Initialize a new CaikitRuntimeServerWrapper

        Args:
            server(grpc.Server): The server that is being wrapped
            global_predict(function): A function that will accept an arbitrary
                gRPC request message and a grpc.ServicerContext, and return
                a suitable gRPC response message
        """

        self._server = server
        self._global_predict = global_predict
        self._intercepted_svc_package = intercepted_svc_package
        self._intercepted_methods = []

        for method in self._intercepted_svc_package.descriptor.methods:
            # Take the method short name (e.g., 'SyntaxIzumoPredict') and
            # concatenate it with the intercepted service name to produce
            # a fully qualified RPC method name that we wish to intercept
            # (e.g., '/natural_language_understanding.CaikitRuntime/SyntaxIzumoPredict')
            fqm = "/%s/%s" % (
                self._intercepted_svc_package.descriptor.full_name,
                method.name,
            )

            log.info("<RUN81194024I>", "Intercepting RPC method %s", fqm)
            self._intercepted_methods.append((method.name, fqm))

    # **************************************************************************
    # Custom methods
    # **************************************************************************

    def intercepted_service(self):
        """Get the fully-qualified name of the intercepted service

        Returns:
            string:
                The fully-qualified name of the service whose RPC handlers are
                intercepted by this server wrapper
        """
        return self._intercepted_svc_package.descriptor.full_name

    def intercepted_methods(self):
        """Get the list of intercepted predict RPC methods

        Returns:
            list((string, string)):
                A list of two-element tuples containing the short name (e.g.,
                'SyntaxIzumoPredict') and fully-qualified name (e.g.,
                '/natural_language_understanding.CaikitRuntime/SyntaxIzumoPredict')
                of every RPC method intercepted by this server wrapper
        """
        return self._intercepted_methods

    @staticmethod
    def safe_rpc_wrapper(rpc: Callable, caikit_rpc: Optional[CaikitRPCBase] = None):
        """This wrapper should be used to safely invoke an RPC. If used, it adds automatic error
        handling and conversion to the appropriate response for gRPC, as well as logging indicating
        if the the error was intentional (i.e., thrown as CaikitRuntimeException directly) or
        unexpected (i.e., thrown as a non GRPC error).

        Args:
            rpc(Function): Method attached to a servicer instance to be invoked in a safe manner.

        Returns:
            A function that takes a gRPC request message and ServicerContext, and safely invokes
            the provided RPC.
        """
        if rpc is None:
            message = "Programming error, RPC is None!"
            log.error("<RUN33322123E>", message)
            raise CaikitRuntimeException(grpc.StatusCode.INTERNAL, message)

        if rpc.__name__ == "safe_rpc_call":
            return rpc
        log.info(
            "<RUN33333123I>",
            "Wrapping safe rpc for %s",
            rpc.__name__,
        )

        def safe_rpc_call(request, context):
            """This function should be used to safely invoke an RPC. If used, it adds automatic
            error handling and conversion to the appropriate response for gRPC, as well as logging
            indicating if the the error was intentional (i.e., thrown as CaikitRuntimeException
            directly) or unexpected (i.e., thrown as a non GRPC error).

            Args:
                request(message.Message): gRPC request object normally received by the rpc.
                context(grpc.ServicerContext): gRPC context object normally received by the rpc.

            Returns:
                gRPC response object return by the invoked RPC or None (aborted context).
            """
            with alog.ContextLog(log.debug, "[Safe RPC]: %s", rpc.__name__):
                try:
                    IN_PROGRESS_GAUGE.labels(rpc_name=rpc.__name__).inc()
                    if caikit_rpc:
                        # Pass through the CaikitRPCBase rpc description to the global handlers
                        return rpc(request, context, caikit_rpc=caikit_rpc)
                    return rpc(request, context)

                except CaikitRuntimeException as e:
                    log_dict = {"log_code": "<RUN89011375W>", "message": e.message}
                    log.warning({**log_dict, **e.metadata})
                    context.abort(e.status_code, e.message)

                except ValueError as e:
                    message = repr(e)
                    log.error("<RUN33333333E>", message)
                    log.error("<RUN33333334E>", str(traceback.format_exc()))
                    context.abort(grpc.StatusCode.UNKNOWN, message)
                finally:
                    IN_PROGRESS_GAUGE.labels(rpc_name=rpc.__name__).dec()

        return safe_rpc_call

    # **************************************************************************
    # Overridden grpc.Server methods
    # **************************************************************************

    def add_generic_rpc_handlers(self, generic_rpc_handlers):
        """Registers GenericRpcHandlers with this Server.

        This method will intercept the generic_rpc_handlers

        Args:
          generic_rpc_handlers: An iterable of GenericRpcHandlers that will be
          used to service RPCs.
        """

        class DummyHandlerCallDetails(grpc.HandlerCallDetails):
            """Dummy class for constructing a grpc.HandlerCallDetails object"""

            def __init__(self, method):
                super().__init__()
                self.method = method

        # Iterate over each grpc.ServiceRpcHandler...
        for handler in generic_rpc_handlers:
            # ...and check if this is the service we wish to intercept
            if handler.service_name() == self.intercepted_service():
                # This is the service whose RPC handlers we wish to intercept
                # and re-route.  We now need to iterate over each method that
                # we wish to re-route, get the original RPC handler for that
                # method (see
                # caikit_runtime_pb2_grpc.add_CaikitRuntimeServicer_to_server
                # for a dict of the rpc_method_handlers we wish to re-route)
                rerouted_rpc_method_handlers = {}
                for method, fqm in self.intercepted_methods():
                    # Get the original grpc.RpcMethodHandler for this RPC method
                    original_rpc_handler = handler.service(DummyHandlerCallDetails(fqm))

                    # Find the Caikit RPC that maps to this rpc
                    caikit_rpc = self._intercepted_svc_package.caikit_rpcs.get(
                        method, None
                    )
                    if not caikit_rpc:
                        raise ValueError(f"No Caikit RPC Found for method: {method}")

                    # Now, swap out the original unary-unary callable with our
                    # generic predict method, and add this newly re-routed RPC
                    # method handler to the dict of (method, handler) pairs
                    safe_rpc_handler = self._make_new_handler(
                        original_rpc_handler, caikit_rpc
                    )
                    rerouted_rpc_method_handlers[method] = safe_rpc_handler
                    log.info(
                        "<RUN30032825I>",
                        "Re-routing RPC %s from %s to %s",
                        fqm,
                        self._get_handler_fn(original_rpc_handler),
                        self._get_handler_fn(rerouted_rpc_method_handlers[method]),
                    )

                # Now that we have re-rerouted all the original RPC method
                # handlers to the global predict RPC method handler, it is time
                # to bind them to the underlying server that we are wrapping
                generic_handler = grpc.method_handlers_generic_handler(
                    self.intercepted_service(), rerouted_rpc_method_handlers
                )
                self._server.add_generic_rpc_handlers((generic_handler,))
                log.info(
                    "<RUN24924908I>",
                    "Interception of service %s complete",
                    self.intercepted_service(),
                )

            else:
                # This is not the service whose RPC handlers we wish to
                # intercept, so just pass the (unmodified) RPC handlers
                # along to the underlying gRPC server we are wrapping
                assert isinstance(handler, grpc._utilities.DictionaryGenericHandler)
                for method in handler._method_handlers:
                    # Wrap the RPC handler for this method in a safe RPC call,
                    # but do not replace the handler with a global handler
                    original_rpc_handler = handler._method_handlers[method]
                    safe_rpc_handler = self._make_new_handler(original_rpc_handler)
                    handler._method_handlers[method] = safe_rpc_handler

                self._server.add_generic_rpc_handlers(generic_rpc_handlers)

    def _make_new_handler(
        self,
        original_rpc_handler: RpcMethodHandler,
        caikit_rpc: Optional[CaikitRPCBase] = None,
    ):
        if caikit_rpc:
            behavior = self.safe_rpc_wrapper(self._global_predict, caikit_rpc)
        else:
            behavior = self.safe_rpc_wrapper(self._get_handler_fn(original_rpc_handler))

        if original_rpc_handler.unary_unary:
            handler_constructor = grpc.unary_unary_rpc_method_handler
        elif original_rpc_handler.unary_stream:
            handler_constructor = grpc.unary_stream_rpc_method_handler
        elif original_rpc_handler.stream_unary:
            handler_constructor = grpc.stream_unary_rpc_method_handler
        else:
            handler_constructor = grpc.stream_stream_rpc_method_handler

        return handler_constructor(
            behavior=behavior,
            request_deserializer=original_rpc_handler.request_deserializer,
            response_serializer=original_rpc_handler.response_serializer,
        )

    # **************************************************************************
    # Pass-through (i.e., unchanged) grpc.Server methods
    # **************************************************************************

    def add_insecure_port(self, address):
        """Opens an insecure port for accepting RPCs.

        This method may only be called before starting the server.

        Args:
          address: The address for which to open a port.
          if the port is 0, or not specified in the address, then gRPC runtime
          will choose a port.

        Returns:
          integer:
          An integer port on which server will accept RPC requests.
        """
        return self._server.add_insecure_port(address)

    def add_secure_port(self, address, server_credentials):
        """Opens a secure port for accepting RPCs.

        This method may only be called before starting the server.

        Args:
          address: The address for which to open a port.
            if the port is 0, or not specified in the address, then gRPC
            runtime will choose a port.
          server_credentials: A ServerCredentials object.
        Returns:
          integer:
          An integer port on which server will accept RPC requests.
        """
        return self._server.add_secure_port(address, server_credentials)

    def start(self):
        """Starts this Server.

        This method may only be called once. (i.e. it is not idempotent).
        """
        self._server.start()

    def stop(self, grace):
        """Stops this Server.

        This method immediately stop service of new RPCs in all cases.

        If a grace period is specified, this method returns immediately
        and all RPCs active at the end of the grace period are aborted.
        If a grace period is not specified (by passing None for `grace`),
        all existing RPCs are aborted immediately and this method
        blocks until the last RPC handler terminates.

        This method is idempotent and may be called at any time.
        Passing a smaller grace value in a subsequent call will have
        the effect of stopping the Server sooner (passing None will
        have the effect of stopping the server immediately). Passing
        a larger grace value in a subsequent call *will not* have the
        effect of stopping the server later (i.e. the most restrictive
        grace value is used).

        Args:
          grace: A duration of time in seconds or None.
        Returns:
          A threading.Event that will be set when this Server has completely
          stopped, i.e. when running RPCs either complete or are aborted and
          all handlers have terminated.
        """
        return self._server.stop(grace)

    def wait_for_termination(self, timeout=None):
        """Block current thread until the server stops.
        The wait will not consume computational resources during blocking,
        and it will block until one of the two following conditions are met:
        1. The server is stopped or terminated;
        2. A timeout occurs if timeout is not None.

        The timeout argument works in the same way as threading.Event.wait().
        Args:
            timeout: A floating point number specifying a timeout for the operation in seconds.
        Returns:
            A bool indicates if the operation times out.
        """
        return self._server.wait_for_termination(timeout)

    @staticmethod
    def _get_handler_fn(handler: RpcMethodHandler) -> Callable:
        if handler.unary_unary:
            return handler.unary_unary
        if handler.unary_stream:
            return handler.unary_stream
        if handler.stream_unary:
            return handler.stream_unary
        return handler.stream_stream
