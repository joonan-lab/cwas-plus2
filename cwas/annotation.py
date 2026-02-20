"""
CWAS Annotation Step

This step annotate user's VCF file using annotation data specified
in the CWAS configuration step. This step mainly uses
Variant Effect Predictor (VEP) to annotate user's VCF file.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from cwas.core.annotation.bed import annotate as _annotate_using_bed
from cwas.core.annotation.vep import VepCmdGenerator
from cwas.runnable import Runnable
from cwas.utils.check import check_is_file
from cwas.utils.check import check_is_dir
from cwas.utils.check import check_num_proc
from cwas.utils.cmd import CmdExecutor, compress_using_bgzip, index_using_tabix
from cwas.utils.log import print_arg, print_log, print_progress

import multiprocessing as mp
from functools import partial
from pysam import TabixFile


class Annotation(Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._vcf_path = None
        self._vep_cmd_gen = None

    @staticmethod
    def _print_args(args: argparse.Namespace):
        print_arg("Target VCF file", args.vcf_path)
        print_arg("Output directory", args.output_dir_path)
        print_arg("Number of worker processes", args.num_proc)
        docker_mode = getattr(args, 'docker_mode', False)
        print_arg("Docker mode", docker_mode)
        if docker_mode:
            vep_ver = getattr(args, 'vep_version', None)
            print_arg("VEP Docker image", f"ensemblorg/ensembl-vep:release_{vep_ver}" if vep_ver else "ensemblorg/ensembl-vep:latest")

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        import re
        check_is_file(args.vcf_path)
        check_is_dir(args.output_dir_path)
        check_num_proc(args.num_proc)
        vep_version = getattr(args, 'vep_version', None)
        docker_mode = getattr(args, 'docker_mode', False)
        if vep_version and not docker_mode:
            raise ValueError("--vep-version requires --docker-mode")
        if vep_version and not re.match(r'^\d+(\.\d+)*$', vep_version):
            raise ValueError(
                f"Invalid --vep-version format: '{vep_version}'. "
                f"Expected numeric version like '115.0'"
            )

    @property
    def docker_mode(self):
        return getattr(self.args, 'docker_mode', False)

    @property
    def vep_version(self):
        return getattr(self.args, 'vep_version', None)

    @property
    def vcf_path(self):
        if self._vcf_path is None:
            if (self.args.num_proc > 1) and (self.args.vcf_path.suffix != '.gz'):
                vcf_gz = compress_using_bgzip(self.args.vcf_path)
                index_using_tabix(vcf_gz)
                self._vcf_path = vcf_gz
            else:
                self._vcf_path = self.args.vcf_path.resolve()

        return self._vcf_path

    @property
    def num_proc(self):
        return self.args.num_proc

    @property
    def output_dir_path(self):
        return self.args.output_dir_path.resolve()

    @property
    def vep_cmd_generator(self):
        if self._vep_cmd_gen is None:
            self._vep_cmd_gen = VepCmdGenerator(
                self.get_env("VEP"), self.get_env("VEP_CACHE_DIR"),
                self.get_env("VEP_CONSERVATION_FILE"), self.get_env("VEP_LOFTEE"),
                self.get_env("VEP_HUMAN_ANCESTOR_FA"), self.get_env("VEP_GERP_BIGWIG"),
                self.get_env("VEP_MIS_DB"), self.get_env("VEP_MIS_INFO_KEY"),
                str(self.vcf_path), str(self.num_proc),
                docker_mode=self.docker_mode,
                vep_version=self.vep_version,
            )
            self._vep_cmd_gen.output_vcf_path = self.vep_output_vcf_path
        return self._vep_cmd_gen

    @property
    def vep_cmd(self):
        return self.vep_cmd_generator.cmd

    @property
    def vep_output_vcf_path(self):
        if self.vcf_path.suffix == '.gz':
            return (
                f"{self.output_dir_path}/"
                f"{self.vcf_path.stem.replace('.vcf', '.vep.vcf')}"
            )
        else:
            return (
                f"{self.output_dir_path}/"
                f"{self.vcf_path.name.replace('.vcf', '.vep.vcf')}"
            )

    @property
    def vep_output_vcf_gz_path(self):
        return self.vep_output_vcf_path.replace(".vcf", ".vcf.gz")

    @property
    def annotated_vcf_path(self):
        return self.vep_output_vcf_gz_path.replace('.vep.vcf.gz', '.annotated.vcf.gz')

    def _check_docker_available(self):
        """Check that docker is available and the VEP image is present."""
        if not shutil.which("docker"):
            raise FileNotFoundError(
                "Docker is not found on PATH. Install Docker or remove --docker-mode."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Docker daemon is not running. Start Docker or remove --docker-mode."
            )
        image = self.vep_cmd_generator.docker_image
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print_progress(f"Pulling Docker image {image}")
            try:
                subprocess.run(["docker", "pull", image], check=True)
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    f"Failed to pull Docker image '{image}'. "
                    f"Check that the VEP version is valid."
                )

    def run(self):
        self.annotate_using_vep()
        self.process_vep_vcf()
        self.annotate_using_bed()
        self.update_env()
        print_progress("Done")

    def annotate_using_vep(self):
        print_progress("Annotation via VEP")
        if (
            Path(self.vep_output_vcf_path).is_file()
            or Path(self.vep_output_vcf_gz_path).is_file()
        ):
            print_log(
                "NOTICE",
                "You have already done the VEP annotations.",
                True,
            )
            return

        if self.docker_mode:
            self._check_docker_available()

        if self.num_proc == 1:
            vep_bin, *vep_args = self.vep_cmd
            CmdExecutor(vep_bin, vep_args).execute_raising_err()
        else:
            self._annotate_multiprocess()

    def _annotate_multiprocess(self):
        print_progress("For multiprocessing, the input VCF should be indexed")
        chroms = self.fetch_chromosomes()

        gen = self.vep_cmd_generator
        output_dir = str(self.output_dir_path)

        multi_inputs = []
        args_list = []
        tmp_output_list = []

        if self.docker_mode:
            docker_prefix = gen.docker_volume_mounts(output_dir)
            vep_option_args = gen.vep_option_args

            for chrom in chroms:
                multi_inputs.append(
                    ' '.join(['tabix -h', str(self.vcf_path), chrom, '|'])
                )
                replace_name = '.' + chrom + '.vep.vcf'
                tmp_output_vcf_path = self.vep_output_vcf_path.replace(".vep.vcf", replace_name)
                tmp_output_filename = os.path.basename(tmp_output_vcf_path)
                container_output = f"{VepCmdGenerator.CONTAINER_OUTPUT_DIR}/{tmp_output_filename}"
                args_list.append([
                    *docker_prefix,
                    gen.docker_image,
                    "vep",
                    "-o", container_output,
                    *vep_option_args,
                ])
                tmp_output_list.append(tmp_output_vcf_path)

            bin_name = "docker"
        else:
            vep_option_args = gen.vep_option_args

            for chrom in chroms:
                multi_inputs.append(
                    ' '.join(['tabix -h', str(self.vcf_path), chrom, '|'])
                )
                replace_name = '.' + chrom + '.vep.vcf'
                tmp_output_vcf_path = self.vep_output_vcf_path.replace(".vep.vcf", replace_name)
                args_list.append(['-o', tmp_output_vcf_path, *vep_option_args])
                tmp_output_list.append(tmp_output_vcf_path)

            bin_name = gen.vep_path

        print_progress(' '.join(["Input VCF has", str(len(chroms)), "number of chromosomes"]))

        num_processes = self.num_proc if self.num_proc < len(chroms) else len(chroms)

        _run_multiple_vep = partial(self.execute_CMD_mp, shell=True)

        pool = mp.Pool(processes=num_processes)
        pool.starmap(
            _run_multiple_vep,
            zip([bin_name for _ in range(len(chroms))], args_list, multi_inputs),
        )
        pool.close()
        pool.join()

        print_progress("Merge and sort output files into a single file")
        args_header = ["'^#'", tmp_output_list[0], ">", self.vep_output_vcf_path]
        CmdExecutor("grep", args_header, shell=True).execute_raising_err()
        args_merge = ["-k1,1V", "-k2,2n", '>>', self.vep_output_vcf_path]
        CmdExecutor("sort", args_merge,
                    multi_input=' '.join(["grep", "--no-filename", "-v", "'^#'", *tmp_output_list, '|']),
                    shell=True).execute_raising_err()
        print_progress("Remove temporary outputs")
        args_remove = [*tmp_output_list]
        CmdExecutor("rm", args_remove).execute_raising_err()

    def execute_CMD_mp(self, bin: str, args: list = [], multi_input: Optional[str] = None, shell: bool = False):
        return CmdExecutor(bin = bin, args = args, multi_input = multi_input, shell=shell).execute_raising_err()

    def fetch_chromosomes(self):
        chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
        chr_list = []
        vcf_reader = TabixFile(str(self.vcf_path))
        for chromosome in chromosomes:
            try:
                vcf_reader.fetch(chromosome)
                chr_list.append(chromosome)
            except StopIteration:
                pass
            except ValueError:
                print_log("LOG", f"Chromosome {chromosome} not found in VCF", True)
        return chr_list

    def process_vep_vcf(self):
        print_progress("Compress the VEP output using bgzip")
        vcf_gz_path = compress_using_bgzip(self.vep_output_vcf_path)
        print_progress("Create an index of the VEP output using tabix")
        index_using_tabix(vcf_gz_path)

    def annotate_using_bed(self):
        print_progress("BED custom annotation")
        if Path(self.annotated_vcf_path).is_file():
            print_log(
                "NOTICE",
                "You have already done the BED custom annotation.",
                True,
            )
            return

        annotate_vcf = _annotate_using_bed(
            self.vep_output_vcf_gz_path,
            self.annotated_vcf_path,
            self.get_env("MERGED_BED"),
            self.num_proc,
        )

        annotate_vcf.bed_custom_annotate()

    def update_env(self):
        self.set_env("ANNOTATED_VCF", self.annotated_vcf_path)
        self.save_env()
