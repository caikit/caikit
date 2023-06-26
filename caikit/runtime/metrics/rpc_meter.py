# Standard
from collections import Counter
import json
import os
import threading
import time
import uuid

# First Party
import alog

# Local
from caikit import get_config

log = alog.use_channel("RPC_METER")


# pylint: disable=too-many-instance-attributes
class RPCMeter:
    """This class contains metering logic for RPC calls affiliated with the Caikit Runtime that are
    not a part of the Model Runtime proto definition.
    """

    def __init__(self):
        """Initialize a RPCMeter instance and start the log writer thread."""

        self.predict_rpc_counter = []
        self.logs_writer_thread = threading.Thread(target=self.write_metrics)
        self.metering_event = threading.Event()
        self.rpc_counter_lock = threading.Lock()
        self.write_file_lock = threading.Lock()
        self.metrics_dir = get_config().runtime.metering.log_dir
        self.unique_id = str(uuid.uuid4()).replace("-", "_")
        self.file_path = os.path.join(
            self.metrics_dir,
            "predict_rpc_metrics_{}.json".format(self.unique_id),
        )
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
        self.logs_writer_thread.start()
        log.debug(
            "<RUN76774000I>",
            "Started metering log writer thread %s",
            self.logs_writer_thread.name,
        )

    def update_metrics(self, model_type):
        """Updates metrics, writes to file if max count has reached and resets counters
        Args:
            model_type (string): Type of model the request was made for
        """
        # Locking to ensure that with concurrent updates to counters, the latest metrics are
        # reported
        with self.rpc_counter_lock:
            self.predict_rpc_counter.append(model_type)

    def end_writer_thread(self):
        """Kills log writer thread"""
        self.metering_event.set()
        if self.logs_writer_thread.is_alive():
            self.logs_writer_thread.join()

    def flush_metrics(self):
        """Writes metrics and kills log writer thread"""
        log.debug(
            "<RUN76774001I>",
            "Server interrupted so flushing metrics to file for thread %s",
            self.logs_writer_thread.name,
        )
        self.end_writer_thread()
        self._write_metrics()

    def write_metrics(self):
        """Function for log writer thread to write logs at specific intervals configured by user"""
        while True:
            log.debug(
                "<RUN76774002I>",
                "Metering log file writing to %s",
                self.file_path,
            )
            self._write_metrics()
            notified = self.metering_event.wait(
                get_config().runtime.metering.log_interval
            )
            if notified:
                log.debug("<RUN76774003I>", "Shutting down metering writer log thread")
                break

    def _write_metrics(self):
        """Writes all metrics to directory specified in config and resets counters"""
        try:
            metrics_dict = {}
            with self.rpc_counter_lock:
                if self.predict_rpc_counter:
                    metrics_dict = {
                        "timestamp": time.time(),
                        "batch_size": len(self.predict_rpc_counter),
                        "model_type_counters": Counter(self.predict_rpc_counter),
                        "container_id": self.unique_id,
                    }
                    self.predict_rpc_counter.clear()
                    log.debug("<RUN76774004I>", "predict_rpc_counter reset")
            if metrics_dict:
                with self.write_file_lock:
                    json_string = json.dumps(metrics_dict)
                    with open(self.file_path, "a", encoding="utf-8") as json_file:
                        json_file.write(json_string + "\n")
                # Log the metrics dict to stdout as well for later scraping if required
                log.info("<RUN76774008I>", metrics_dict)
                log.info(
                    "<RUN76774005I>",
                    "Successfully written metrics file to %s",
                    self.file_path,
                )
            else:
                log.info(
                    "<RUN76774006I>", "No new RPCs to write, skipping metering logging"
                )
        except json.JSONDecodeError as e:
            log.info("<RUN76774007I>", "Write metrics failed with %s", str(e))
        except FileNotFoundError as e:
            log.info("<RUN76774007I>", "Write metrics failed with %s", str(e))

    def __del__(self):
        self.end_writer_thread()
