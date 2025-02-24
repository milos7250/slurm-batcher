#!/usr/bin/env python
# %%
import argparse
import json
import logging
import re
import shlex
import stat
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from html import parser
from pathlib import Path
from time import sleep

import pandas as pd
from simple_slurm import Slurm


class CustomFormatter(logging.Formatter):
    teal = "\033[36;20m"
    yellow = "\033[33;20m"
    red = "\033[31;20m"
    bold_red = "\033[31;1m"
    reset = "\033[0m"

    green = "\033[32;20m"
    blue = "\033[34;20m"
    format1 = green + "[%(asctime)s] " + blue + "%(name)s" + reset + ": "
    format2 = " - %(message)s"

    FORMATS = {
        logging.DEBUG: format1 + teal + "%(levelname)s" + reset + format2,
        logging.INFO: format1 + teal + "%(levelname)s" + reset + format2,
        logging.WARNING: format1 + yellow + "%(levelname)s" + reset + format2,
        logging.ERROR: format1 + red + "%(levelname)s" + reset + format2,
        logging.CRITICAL: format1 + bold_red + "%(levelname)s" + reset + format2,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


log = logging.getLogger("slurm-batcher")
log.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
for handler in log.handlers:
    handler.close()
    log.removeHandler(handler)
log.addHandler(handler)


def set_log_level(level):
    log.setLevel(level)
    handler.setLevel(level)


@contextmanager
def with_log_level(level):
    old_level = log.getEffectiveLevel()
    try:
        set_log_level(level)
        yield
    finally:
        set_log_level(old_level)


set_log_level(logging.INFO)


class ARGUMENT(str):
    pass


class MISSING(ARGUMENT):
    def __repr__(self):
        return "MISSING"


class USED(ARGUMENT):
    def __repr__(self):
        return f"USED: {super().__repr__()}"


class UNUSED(ARGUMENT):
    def __repr__(self):
        return f"UNUSED: {super().__repr__()}"


class Command:
    def __init__(self, cmd: str, args: dict = None):
        self.cmd = cmd
        self.args = {}
        for part in re.split(r"[{}]", self.cmd)[1::2]:
            self.args[part] = MISSING()
        if args:
            for k, v in args.items():
                if not isinstance(v, MISSING) or (not isinstance(v, MISSING) and v == ""):
                    if k in self.args:
                        self.args[k] = USED(v)
                    else:
                        self.args[k] = UNUSED(v)
        log.debug(f"Created command {self}")

    def update_args(self, args: dict):
        return Command(self.cmd, self.args | args)

    def cast_command(self, metadata: list[dict]):
        log.debug(f"Casting command {self.cmd} with metadata {metadata}")
        return [self.update_args(row) for row in metadata]

    @property
    def has_missing(self):
        return any([isinstance(arg, MISSING) for arg in self.args.values()])

    @property
    def has_unused(self):
        return any([isinstance(arg, UNUSED) for arg in self.args.values()])

    def __call__(self):
        if self.has_missing:
            log.critical(
                f"Command {self.cmd} has missing args { {k: v for k, v in self.args.items() if isinstance(v, MISSING)} }"
            )
            sys.exit(1)
        if self.has_unused:
            # log.debug(f"Command {self.cmd} has unused args {self.args}")
            pass
        return self.cmd.format(**self.args)

    def __repr__(self):
        return f"Command {self.cmd} with args { {k: v for k, v in self.args.items() if not isinstance(v, UNUSED)} }"


def collate(args, slurm_args=None):
    # Make sure job dependencies are satisfied
    if args["dependencies"]:
        with with_log_level(logging.ERROR):
            metadata = status(args, slurm_args)
        while not all(metadata["status"].eq("COMPLETED")):
            if any(metadata["status"].isin(["OUT_OF_ME+", "FAILED", "CANCELLED"])):
                log.critical("Job dependencies could not be satisfied")
                sys.exit(4)
            log.warning("All dependencies are not completed, waiting for completion...")
            sleep(60)
            with with_log_level(logging.ERROR):
                metadata = status(args, slurm_args)

    metadata: pd.DataFrame = args["metadata"]
    del args["metadata"]
    metadata["output_dir"] = args["output_dir"]
    cmd = Command(args["command"])

    # Collate the arguments from metadata, handling missing values
    metadata = metadata.reset_index()[cmd.args.keys()]
    nunique = metadata.nunique()
    collate_fields = nunique[nunique > 1].index
    cmd_args = metadata.iloc[0].to_dict()

    for k, v in metadata[collate_fields].to_dict(orient="list").items():
        is_missing = [isinstance(i, MISSING) or i == "" for i in v]
        if not any(is_missing):
            cmd_args[k] = " ".join(v)
        elif all(is_missing):
            cmd_args[k] = MISSING()
        else:
            non_missing = [i for i, missing in zip(v, is_missing) if not missing]
            cmd_args[k] = " ".join(non_missing)
            log.warning(f"Collated {k} with {len(v) - len(non_missing)} missing values")

    cmd = cmd.update_args(cmd_args)
    log.info(f"Generated command {cmd()}")
    if not args["dry_run"]:
        process = subprocess.Popen(f"exec bash -c '{cmd()}'", shell=True)
        process.wait()


def status(args, slurm_args=None):
    metadata: pd.DataFrame = args["metadata"]
    for index in metadata.index:
        if metadata.loc[index, "job_id"] is not None:
            res = (
                subprocess.check_output(
                    ["sacct", "--units=G", "-j", metadata.loc[index, "job_id"], "-o", "state,elapsed,MaxRSS", "-n"]
                )
                .decode()
                .split("\n")
            )
            if len(res) > 2:
                res = res[1].strip().split()
            else:
                res = res[0].strip().split()
            if len(res) == 2:
                res.append("N/A")
            metadata.loc[index, ["status", "time_elapsed", "mem_used"]] = res
            log.info(
                f"Job {metadata.loc[index, 'job_id']} is {metadata.loc[index, 'status']} for {metadata.loc[index, 'time_elapsed']} with {metadata.loc[index, 'mem_used']} memory"
            )
    if args["save_status"]:
        metadata.to_csv(args["metadata_file"])
    return metadata


def rerun(args, slurm_args=None):
    if "--mem" not in slurm_args:
        log.critical("Rerun requires a memory limit to be set")
        sys.exit(3)
    with with_log_level(logging.ERROR):
        metadata = status(args, slurm_args)
    for index in metadata[metadata["status"] == "OUT_OF_ME+"].index:
        script = (
            Path(args["metadata_file"]).parent
            / "scripts"
            / f"{metadata.loc[index, 'command'].split(' ')[0]}-{metadata.loc[index, 'job_id']}.sh"
        )
        if not script.exists():
            log.critical(f"Script {script} could not be found")
            sys.exit(4)
        out = subprocess.check_output(["sbatch", *slurm_args, script])
        metadata.loc[index, "job_id"] = re.search("[0-9]+", out.decode())[0]
        script.rename(
            Path(args["metadata_file"]).parent
            / "scripts"
            / f"{metadata.loc[index, 'command'].split(' ')[0]}-{metadata.loc[index, 'job_id']}.sh"
        )
        log.info(f"Resubmitted job {metadata.loc[index, 'job_id']} with script {script} and {' '.join(slurm_args)}")
    metadata.to_csv(args["metadata_file"])


def run(args, slurm_args=[]):
    metadata: pd.DataFrame = args["metadata"]
    del args["metadata"]

    if (config := Path(sys.argv[0].replace(".py", ".conf"))).exists():
        log.info(f"Using config file {config}")
        with open(config, "r") as f:
            config = json.load(f)
            [slurm_args.extend(i) for i in config["slurm_args"].items()]
    if args["add_to_metadata"] and len(args["add_to_metadata"]) % 2 != 0:
        log.critical("add-to-metadata must have an even number of arguments")
        sys.exit(2)
    slurm_args = [re.sub("(?<!-)-(?!-)", "_", arg) for arg in slurm_args]
    log.info(f"Parsed args\n{textwrap.indent(json.dumps(args, indent=2), ' ' * 48)}")
    log.info(f"Parsed slurm args {slurm_args}")

    # Use the temporary directory if requested
    args["real_output_dir"] = args["output_dir"]
    if args["use_tmpdir"]:
        args["output_dir"] = f"$TMPDIR/{args['output_dir']}"
        log.info(f"Using temporary directory {args['output_dir']} for scripts")

    # Load the metadata
    metadata["output_dir"] = args["output_dir"]
    metadata["real_output_dir"] = args["real_output_dir"]
    metadata["conda_env_name"] = args["conda_env_name"]

    # Cast commads to the metadata
    batch = Command(args["command"]).cast_command(metadata.reset_index().to_dict(orient="records"))
    log.info(f"Generated {len(batch)} commands")

    # Create the SLURM jobs
    for i, cmd in enumerate(batch):
        slurm = Slurm(*slurm_args)
        slurm.add_arguments(output=f"{cmd.args['real_output_dir']}/logs/{Slurm.JOB_NAME}-{Slurm.JOB_ID}.log")
        if "--job-name" not in slurm_args:
            slurm.add_arguments(job_name=cmd.cmd.split()[0])

        # Handle dependencies
        if args["dependencies"]:
            if "job_id" not in cmd.args or cmd.args["job_id"] == "":
                log.critical(f"Command {cmd} does not have a job_id value in metadata")
                sys.exit(1)
            job_state = json.loads(
                subprocess.check_output(
                    ["sacct", "-j", cmd.args["job_id"], "--json"],
                ).decode()
            )["jobs"][0]["state"]["current"][0]
            if job_state in ["PENDING", "RUNNING"]:
                slurm.add_arguments(dependency=f"afterok:{cmd.args['job_id']}")
            elif job_state != "COMPLETED":
                log.error(f"Dependency of {cmd} is in state {job_state}, cannot proceed.")
                continue

        # Add pre commands
        slurm.add_cmd(f"mkdir -p {cmd.args['output_dir']}")
        for pre_cmd in config["pre_command"]:
            slurm.add_cmd(Command(pre_cmd, cmd.args)())
        # Add the main command
        slurm.add_cmd(cmd())
        # Add post commands
        for post_cmd in config["post_command"]:
            slurm.add_cmd(Command(post_cmd, cmd.args)())
        if args["use_tmpdir"]:
            slurm.add_cmd(f"rsync -avP {cmd.args['output_dir']}/ {cmd.args['real_output_dir']}")
        log.debug(f"Generated script:\n{slurm.script(shell='/usr/bin/bash', convert=False)}")

        # Create the script and submit the job
        Path(f"{cmd.args['real_output_dir']}/scripts").mkdir(exist_ok=True, parents=True)
        cmd.args["job_id"] = i
        if not args["dry_run"]:
            try:
                cmd.args["job_id"] = slurm.sbatch(convert=True, shell="/usr/bin/bash", verbose=False)
                log.debug(f"Submitted job {cmd.args['job_id']}")
            except AssertionError:
                log.critical(f"Failed to submit job {cmd}")
                with open(
                    f"{cmd.args['real_output_dir']}/scripts/{cmd.cmd.split()[0]}-{cmd.args['job_id']}.sh", "w"
                ) as f:
                    f.write(slurm.script(shell="/usr/bin/bash", convert=False))
                sys.exit(5)
        with open(f"{cmd.args['real_output_dir']}/scripts/{cmd.cmd.split()[0]}-{cmd.args['job_id']}.sh", "w") as f:
            f.write(slurm.script(shell="/usr/bin/bash", convert=False))
    if not args["dry_run"]:
        log.info(f"Submitted {len(batch)} jobs")

    # Update metadata fields
    metadata["output_dir"] = args["real_output_dir"]
    metadata["command"] = [cmd() for cmd in batch]
    metadata["job_id"] = [cmd.args["job_id"] for cmd in batch]
    if args["add_to_metadata"]:
        for k, v in zip(args["add_to_metadata"][::2], args["add_to_metadata"][1::2]):
            metadata[k] = [cmd() for cmd in Command(v).cast_command(metadata.reset_index().to_dict(orient="records"))]
    del metadata["conda_env_name"]
    del metadata["output_dir"]
    del metadata["real_output_dir"]

    # Save the metadata
    Path(args["real_output_dir"]).mkdir(exist_ok=True, parents=True)
    if (Path(args["real_output_dir"]) / Path(args["metadata_file"]).name).exists():
        metadata_old = pd.read_csv(
            Path(args["real_output_dir"]) / Path(args["metadata_file"]).name, index_col="sample_name"
        )
        metadata.combine_first(metadata_old).to_csv(Path(args["real_output_dir"]) / Path(args["metadata_file"]).name)
    else:
        metadata.to_csv(Path(args["real_output_dir"]) / Path(args["metadata_file"]).name)
    log.info(f"Saved metadata to {Path(args['real_output_dir']) / Path(args['metadata_file']).name}")

    # Save the script command
    with open(Path(args["real_output_dir"]) / "command.sh", "w") as f:
        f.write(f"#!/usr/bin/env bash\n\n'{"' '".join(sys.argv)}'")
    log.info(f"Saved script command to {Path(args['real_output_dir']) / 'command.sh'}")


def parse_arguments(args):
    # Create the root parser
    parser_root = argparse.ArgumentParser(description="Run a command on a SLURM cluster")
    subparsers = parser_root.add_subparsers()
    parser_root.add_argument(
        "-m", "--metadata", type=str, help="Path to the metadata file", required=True, dest="metadata_file"
    )
    parser_root.add_argument("--head", type=int, help="Only process the first n rows of the metadata file")
    parser_root.add_argument("--tail", type=int, help="Only process the last n rows of the metadata file")
    parser_root.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Do not run jobs, save the generated slurm scripts, or print the commands instead",
        required=False,
    )
    parser_root.add_argument("-v", "--verbose", action="store_true", help="Increase the verbosity of the logger")

    # Create the subparser for status
    parser_status = subparsers.add_parser("status", help="Get the status of the jobs in the metadata file")
    parser_status.set_defaults(func=status)

    # Create the subparser for rerun
    parser_rerun = subparsers.add_parser("rerun", help="Rerun jobs that failed in the metadata file")
    parser_rerun.set_defaults(func=rerun)

    # Create the subparser for collate
    parser_collate = subparsers.add_parser("collate", help="Run a command with columns aggregated into one string")
    parser_collate.add_argument("-c", "--command", type=str, help="Command to run", required=True)
    parser_collate.add_argument("-o", "--output-dir", type=str, help="Output directory", required=True)
    parser_collate.add_argument("--dependencies", action="store_true", help="Include job dependencies", required=False)
    parser_collate.set_defaults(func=collate)

    # Create the subparser for run
    parser_run = subparsers.add_parser("run", help="Run a command on a SLURM cluster")
    parser_run.add_argument("-c", "--command", type=str, help="Command to run", required=True)
    parser_run.add_argument("-o", "--output-dir", type=str, help="Output directory", required=True)
    parser_run.add_argument("-e", "--conda-env-name", type=str, help="Conda environment name", required=False)
    parser_run.add_argument("-j", "--job-name", type=str, help="Job name", required=False)
    parser_run.add_argument(
        "-a",
        "--add-to-metadata",
        type=str,
        help="Add a column to the metadata file",
        required=False,
        nargs="+",
        metavar="KEY VALUE",
    )
    parser_run.add_argument(
        "-t",
        "--use-tmpdir",
        action="store_true",
        help="Use a temporary directory on the processing node",
        required=False,
    )
    parser_run.add_argument(
        "-d",
        "--dependencies",
        action="store_true",
        help="Include job dependencies based on job_id column",
        required=False,
    )
    parser_run.set_defaults(func=run)

    # Parse the arguments
    args, slurm_args = parser_root.parse_known_args(args)
    return args, slurm_args


# %%
if __name__ == "__main__":
    args = sys.argv[1:]
    args, slurm_args = parse_arguments(args)
    func = args.func
    del args.func
    if args.verbose:
        set_log_level(logging.DEBUG)
    save_status = not (args.head or args.tail)

    # Load metadata, process head/tail, and convert to dict
    metadata = pd.read_csv(args.metadata_file, index_col="sample_name", dtype=str)
    if args.head:
        metadata = metadata.head(args.head)
        if args.head > 0:
            log.info(f"Processing the first {args.head} entries of the metadata file")
        else:
            log.info(f"Discarding the last {-args.head} entries of the metadata file")
        del args.head
    if args.tail:
        metadata = metadata.tail(args.tail)
        if args.tail > 0:
            log.info(f"Processing the last {args.tail} entries of the metadata file")
        else:
            log.info(f"Discarding the first {-args.tail} entries of the metadata file")
        del args.tail
    args.metadata = metadata

    args = vars(args)
    args["save_status"] = save_status
    func(args, slurm_args)

# %%
