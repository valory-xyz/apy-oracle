# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Script for generating deployment environments."""
import json
import os
from typing import Dict, List

from deployments.base_deployments import BaseDeployment
from deployments.constants import DEPLOYMENT_REPORT
from deployments.generators.docker_compose.docker_compose import DockerComposeGenerator
from deployments.generators.kubernetes.kubernetes import KubernetesGenerator


AGENTS: Dict[str, str] = {
    "oracle_hardhat": "./deployments/deployment_specifications/price_estimation_hardhat.yaml",
    "oracle_ropsten": "./deployments/deployment_specifications/price_estimation_ropsten.yaml",
    "apy_hardhat": "./deployments/deployment_specifications/apy_estimation_hardhat.yaml",
    "price_estimation_hardhat": "./deployments/deployment_specifications/price_estimation_ropsten.yaml",
}


DEPLOYMENT_OPTIONS = {
    "kubernetes": KubernetesGenerator,
    "docker-compose": DockerComposeGenerator,
}


def generate_deployment(
    type_of_deployment: str,
    valory_application: str,
    configure_tendermint: bool,
) -> str:
    """Generate the deployment build for the valory app."""
    deployment_generator = DEPLOYMENT_OPTIONS[type_of_deployment]
    app_instance = BaseDeployment(
        path_to_deployment_spec=AGENTS[valory_application]  # update in aea.
    )
    deployment = deployment_generator(deployment_spec=app_instance)
    deployment.generate(app_instance)  # type: ignore
    run_command = deployment.generate_config_tendermint(app_instance)  # type: ignore
    deployment.write_config()
    report = DEPLOYMENT_REPORT.substitute(
        **{
            "app": valory_application,
            "type": type_of_deployment,
            "agents": app_instance.number_of_agents,
            "network": app_instance.network,
            "size": len(deployment.output),
        }
    )
    if type_of_deployment == DockerComposeGenerator.deployment_type:
        if configure_tendermint:
            res = os.popen(run_command)  # nosec:
            print(res.read())
        else:
            print(
                f"To configure tendermint for deployment please run: \n\n{run_command}"
            )
    elif type_of_deployment == KubernetesGenerator.deployment_type:
        if configure_tendermint:
            deployment.write_config(run_command)  # type:ignore
        else:
            print("To configure tendermint please run generate and run a config job.")
    return report


def read_keys(file_path: str) -> List[str]:
    """Read in keys from a file on disk."""
    with open(file_path, "r", encoding="utf8") as f:
        keys = json.loads(f.read())
    for key in keys:
        assert "address" in key.keys(), "Key file incorrectly formatted."
        assert "private_key" in key.keys(), "Key file incorrectly formatted."
    return [f["private_key"] for f in keys]
