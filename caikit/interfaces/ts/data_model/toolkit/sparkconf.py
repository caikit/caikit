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
"""Defines function(s) for obtaining a spark configurations."""

# Standard
from typing import Union
import socket

# Local
from .optional_dependencies import HAVE_PYSPARK

WE_HAVE_PYSPARK = HAVE_PYSPARK

if WE_HAVE_PYSPARK:
    # Third Party
    from pyspark import SparkConf


def sparkconf_local(
    master: str = "local[2]",
    executor_memory: str = "2g",
    driver_memory: str = "2g",
    app_name: str = "unnamed",
    **kwargs,
):
    """Returns a SparkConf object configured for spark-local operation

    Args:
        executor_memory (str, optional): Exectuor memory. Defaults to "2g".
        driver_memory (str, optional): Driver memory. Defaults to "2g".
        app_name (str, optional): Spark application name. Defaults to "unnamed".
        kwargs: passthru key,value arguments that will be added to the spark configuration

    Returns:
        SparkConf: a spark configuration object.
    """

    if not WE_HAVE_PYSPARK:
        return {}

    if master.find("local[") != 0:
        raise ValueError(
            "master for local session must be in form 'local[N]' where N is either an integer or *"
        )

    return sparkconf_k8s(
        master=master,
        executor_memory=executor_memory,
        driver_memory=driver_memory,
        app_name=app_name,
        namespace="foo",
        driver_image="foo",
        executor_image="foo",
        **kwargs,
    )


# pylint: disable=line-too-long
def sparkconf_k8s(
    app_name: str,
    namespace: str,
    executor_image: str,
    driver_image: str,
    master: str = "k8s://https://kubernetes.default.svc:443",
    num_executors: str = "2",
    executor_memory: str = "1g",
    executor_cores: str = "2",
    driver_memory: str = "1g",
    driver_cores: str = "2",
    pvc_mount_path: Union[str, None] = None,
    pvc_claim_name: Union[str, None] = None,
    python_path: Union[str, None] = None,
    k8s_service_account: Union[str, None] = None,
    **kwargs,
):
    """Return a spark configuraion object for use on a kubernetes cluster. For more information on
    what some of these parameters are for see
    https://spark.apache.org/docs/latest/running-on-kubernetes.html

    NOTE: if you are simply running a local spark job, we advise you use the sparkconf_local method
    instead as it has fewer parameters and more defaults to get you going more quickly.

    Args:
        app_name (str): The application name (useful for for keeping track of jobs on a multiuser
            cluster)
        namespace (str): k8s namespace in which this job will run (e.g., "default")
        executor_image (str): The container image to use for spark executors.
        driver_image (str): The spark driver image to use (tpyically the same as exectuor image)
        master (_type_, optional): The master specificication. Defaults to
            "k8s://https://kubernetes.default.svc:443".
        num_executors (str, optional): The number of executors to run. Defaults to "2".
        executor_memory (str, optional): The maximum memory allocated to each executor (use g or M
            notation). Defaults to "1g".
        executor_cores (str, optional): The maximum number of cores per executor. Defaults to "2".
        driver_memory (str, optional): The maxumum memory allocated to the driver. Defaults to
            "1g".
        driver_cores (str, optional): The maximum number of cores allocated to the driver. Defaults
            to "2".
        pvc_mount_path (str | None, optional): The PVC mount path for exectuors and driver to mount
            (this usually has to be rwX). Defaults to None.
        pvc_claim_name (str | None, optional): The PVC claim name assocated with the PVC mount.
            Defaults to None.
        python_path (str | None, optional): The python path to use in python jobs in executor and
            driver python processes. Defaults to None.
        k8s_service_account (str | None, optional): The k8s service account to use. Defaults to
            None.
        kwargs: passthru key,value arguments that will be added to the spark configuration

    Returns:
        SparkConf: A spark configuration that has been defined in a way that makes it compatible
            with time series use cases and intended for use with a k8s cluster.
    """

    if not WE_HAVE_PYSPARK:
        return {}

    conf: SparkConf = (
        SparkConf().setAppName(f"{app_name}.caikit.{namespace}").setMaster(master)
    )

    # pushing config out of global configuration file
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # executor/driver spec
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.executor.memory", executor_memory)
    conf.set("spark.executor.cores", executor_cores)
    conf.set("spark.driver.cores", driver_cores)
    conf.set("spark.executor.instances", num_executors)
    conf.set("spark.sql.session.timeZone", "UTC")

    # kubernetes specific
    if "K8S" in master.upper():
        if python_path:
            conf.setExecutorEnv("PYTHONPATH", python_path)
        conf.set("spark.kubernetes.namespace", namespace)
        conf.set("spark.kubernetes.executor.container.image", executor_image)
        conf.set(
            "spark.kubernetes.driver.container.image",
            driver_image if driver_image else executor_image,
        )
        conf.set("spark.kubernetes.driver.annotation.sidecar.istio.io/inject", "false")
        conf.set(
            "spark.kubernetes.executor.annotation.sidecar.istio.io/inject", "false"
        )
        # networking minutia
        conf.set("spark.driver.host", socket.gethostbyname(socket.gethostname()))
        conf.set("spark.driver.port", "37371")
        conf.set("spark.blockManager.port", "6060")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # conf.set("spark.kubernetes.authenticate.driver.serviceAccountName", "spark")

        if pvc_mount_path:
            conf.set(
                f"spark.kubernetes.executor.volumes.persistentVolumeClaim.{pvc_claim_name}.mount.path",
                pvc_mount_path,
            )
        conf.set(
            f"spark.kubernetes.executor.volumes.persistentVolumeClaim.{pvc_claim_name}.mount.readOnly",
            "false",
        )

        if pvc_claim_name:
            conf.set(
                f"spark.kubernetes.executor.volumes.persistentVolumeClaim.{pvc_claim_name}.options.claimName",
                pvc_claim_name,
            )

        if k8s_service_account:
            conf.set(
                "spark.kubernetes.authenticate.driver.serviceAccountName",
                k8s_service_account,
            )

        conf.set("spark.kubernetes.container.image.pullPolicy", "Always")

    for param, val in kwargs.items():
        conf.set(param, val)

    return conf
