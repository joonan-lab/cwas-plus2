"""
Command line generator for Variant Effect Predictor (VEP)
"""
import os
import shutil
import subprocess
import warnings

from cwas.utils.check import check_is_file
from cwas.utils.check import check_is_dir


class VepCmdGenerator:
    DOCKER_REPO = "ensemblorg/ensembl-vep"
    CONTAINER_VEP_DIR = "/opt/vep/.vep"
    CONTAINER_INPUT_DIR = "/input"
    CONTAINER_OUTPUT_DIR = "/output"

    def __init__(self, vep_path: str,
                 vep_cache_path: str, vep_conservation_path: str,
                 vep_loftee_path: str, vep_human_ancestor_fa_path: str,
                 vep_gerp_bw_path: str, vep_mis_db_path: str,
                 vep_mis_info_key: str, input_vcf_path: str,
                 num_proc: str, docker_mode: bool = False,
                 vep_version: str = None) -> None:
        self._vep_path = vep_path if vep_path else ""
        self._vep_cache_path = vep_cache_path
        self._vep_conservation_path = vep_conservation_path
        self._vep_loftee_path = vep_loftee_path
        self._vep_human_ancestor_fa_path = vep_human_ancestor_fa_path
        self._vep_gerp_bw_path = vep_gerp_bw_path
        self._vep_mis_db_path = vep_mis_db_path
        self._vep_mis_info_key = vep_mis_info_key
        self._input_vcf_path = input_vcf_path
        self._docker_mode = docker_mode
        self._vep_version = vep_version
        self._check_validity()
        self._output_vcf_path = input_vcf_path.replace(".vcf", ".vep.vcf")
        self._num_proc = num_proc

    @staticmethod
    def _check_path(path: str, message: str, is_dir: bool = False):
        try:
             if is_dir:
                 check_is_dir(path)
             else:
                 check_is_file(path)
        except ValueError:
            raise ValueError(f"{message}: {path}")
        except Exception:
            raise

    def _check_validity(self):
         if not self._docker_mode:
             self._check_path(self._vep_path, "Invalid VEP path")
         self._check_path(self._vep_conservation_path, "Invalid VEP resource path (conservation file)")
         self._check_path(self._vep_loftee_path, "Invalid VEP resource path (loftee directory)", is_dir=True)
         self._check_path(self._vep_human_ancestor_fa_path, "Invalid VEP resource path (human ancestor fasta file)")
         self._check_path(self._vep_gerp_bw_path, "Invalid VEP resource path (gerp bigwig file)")
         self._check_path(self._vep_mis_db_path, "Invalid VEP resource path (missense database file)")
         self._check_path(self._input_vcf_path, "Invalid input VCF path")
         self._check_path(self._vep_cache_path, "Invalid VEP cache directory path", is_dir=True)

    def _translate_path(self, host_path: str) -> str:
        """Replace vep_cache_path prefix with container mount point."""
        cache_prefix = self._vep_cache_path.rstrip('/') + '/'
        if not host_path.startswith(cache_prefix):
            raise ValueError(
                f"Resource path '{host_path}' is not under VEP cache dir "
                f"'{self._vep_cache_path}'. Cannot translate to container path."
            )
        return self.CONTAINER_VEP_DIR + '/' + host_path[len(cache_prefix):]

    @property
    def docker_mode(self) -> bool:
        return self._docker_mode

    @property
    def docker_image(self) -> str:
        if self._vep_version:
            return f"{self.DOCKER_REPO}:release_{self._vep_version}"
        return f"{self.DOCKER_REPO}:latest"

    @property
    def vep_path(self) -> str:
        return self._vep_path

    @property
    def vep_conservation_path(self) -> str:
        return self._vep_conservation_path

    @property
    def vep_loftee_path(self) -> str:
        return self._vep_loftee_path

    @property
    def vep_human_ancestor_fa_path(self) -> str:
        return self._vep_human_ancestor_fa_path

    @property
    def vep_gerp_bw_path(self) -> str:
        return self._vep_gerp_bw_path

    @property
    def vep_mis_db_path(self) -> str:
        return self._vep_mis_db_path

    @property
    def vep_mis_info_key(self) -> str:
        return self._vep_mis_info_key

    @property
    def vep_cache_path(self) -> str:
        return self._vep_cache_path

    @property
    def input_vcf_path(self) -> str:
        return self._input_vcf_path

    @property
    def output_vcf_path(self) -> str:
        return self._output_vcf_path

    @output_vcf_path.setter
    def output_vcf_path(self, arg: str):
        self._output_vcf_path = arg

    @property
    def num_proc(self) -> str:
        return self._num_proc

    @property
    def cmd_str(self) -> str:
        return " ".join(self.cmd)

    @property
    def vep_option_args(self) -> list:
        """Return VEP option args (no binary, no -i/-o).
        In docker mode, paths are translated to container paths."""
        if self._docker_mode:
            return (self.docker_vep_option_args
                    + self.cmd_option_pick_one_gene_isoform
                    + self.cmd_option_pick_nearest_gene)
        else:
            return (self.cmd_option_basic
                    + self.cmd_option_pick_one_gene_isoform
                    + self.cmd_option_pick_nearest_gene)

    @property
    def cmd(self) -> list:
        if self._docker_mode:
            return self._docker_cmd()
        result = [
            self.vep_path,
            "-i",
            self.input_vcf_path,
            "-o",
            self.output_vcf_path,
        ]
        result += self.cmd_option_basic
        result += self.cmd_option_pick_one_gene_isoform
        result += self.cmd_option_pick_nearest_gene
        return result

    # System libraries that exist in the container and must NOT be
    # overridden by the host versions (glibc mismatch causes crashes).
    _SKIP_LIBS = frozenset([
        "libc.so", "libm.so", "libz.so", "libpthread.so", "libdl.so",
        "librt.so", "libnsl.so", "libresolv.so", "ld-linux",
    ])

    @classmethod
    def _samtools_volume_mounts(cls) -> list:
        """Return -v flags to mount the host samtools binary and its
        shared libraries into the container.  Returns an empty list if
        samtools is not found on the host.  Common system libraries
        (libc, libm, etc.) are skipped to avoid glibc conflicts."""
        samtools = shutil.which("samtools")
        if not samtools:
            warnings.warn(
                "samtools not found on the host. "
                "LOFTEE requires samtools for ancestral allele lookup. "
                "Install samtools (e.g. 'apt install samtools' or "
                "'conda install -c bioconda samtools') to enable full "
                "LOFTEE functionality in Docker mode.",
                stacklevel=2,
            )
            return []
        mounts = ["-v", f"{samtools}:{samtools}"]
        try:
            ldd_out = subprocess.check_output(
                ["ldd", samtools], text=True, stderr=subprocess.DEVNULL,
            )
            for line in ldd_out.splitlines():
                parts = line.split()
                # lines like:  libhts.so.3 => /lib/.../libhts.so.3 (0x...)
                if "=>" in parts and len(parts) >= 3:
                    lib_path = parts[parts.index("=>") + 1]
                    if not lib_path.startswith("/"):
                        continue
                    lib_name = os.path.basename(lib_path)
                    if any(skip in lib_name for skip in cls._SKIP_LIBS):
                        continue
                    mounts += ["-v", f"{lib_path}:{lib_path}"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return mounts

    def _docker_cmd(self) -> list:
        """Build docker run command for VEP."""
        uid = os.getuid()
        gid = os.getgid()
        input_dir = str(os.path.dirname(self._input_vcf_path))
        output_dir = str(os.path.dirname(self._output_vcf_path))
        input_filename = os.path.basename(self._input_vcf_path)
        output_filename = os.path.basename(self._output_vcf_path)

        result = [
            "docker", "run", "--rm",
            "--user", f"{uid}:{gid}",
            "-v", f"{self._vep_cache_path}:{self.CONTAINER_VEP_DIR}",
            "-v", f"{input_dir}:{self.CONTAINER_INPUT_DIR}",
            "-v", f"{output_dir}:{self.CONTAINER_OUTPUT_DIR}",
        ]
        result += self._samtools_volume_mounts()
        result += [
            self.docker_image,
            "vep",
            "-i", f"{self.CONTAINER_INPUT_DIR}/{input_filename}",
            "-o", f"{self.CONTAINER_OUTPUT_DIR}/{output_filename}",
        ]
        result += self.docker_vep_option_args
        result += self.cmd_option_pick_one_gene_isoform
        result += self.cmd_option_pick_nearest_gene
        return result

    def docker_volume_mounts(self, output_dir: str) -> list:
        """Return docker run prefix args (used in multi-process mode)."""
        uid = os.getuid()
        gid = os.getgid()
        result = [
            "run", "--rm", "-i",
            "--user", f"{uid}:{gid}",
            "-v", f"{self._vep_cache_path}:{self.CONTAINER_VEP_DIR}",
            "-v", f"{output_dir}:{self.CONTAINER_OUTPUT_DIR}",
        ]
        result += self._samtools_volume_mounts()
        return result

    @property
    def docker_vep_option_args(self) -> list:
        """Return VEP options with container-translated paths."""
        return [
            "--assembly",
            "GRCh38",
            "--offline",
            "--cache",
            "--dir_cache",
            self.CONTAINER_VEP_DIR,
            "--force_overwrite",
            "--format",
            "vcf",
            "--vcf",
            "--no_stats",
            "--plugin",
            ','.join(['LoF',
                      'conservation_file:' + self._translate_path(self.vep_conservation_path),
                      'loftee_path:' + self._translate_path(self.vep_loftee_path),
                      'human_ancestor_fa:' + self._translate_path(self.vep_human_ancestor_fa_path),
                      'gerp_bigwig:' + self._translate_path(self.vep_gerp_bw_path)]),
            "--dir_plugins",
            self._translate_path(self.vep_loftee_path),
            "--custom",
            ",".join([self._translate_path(self._vep_mis_db_path), "MisDb", 'vcf', "exact", "0", self.vep_mis_info_key]),
        ]

    @property
    def cmd_option_basic(self) -> list:
        """Return basic options (no plugins) of VEP"""
        return [
            "--assembly",
            "GRCh38",
            "--offline",
            "--cache",
            "--dir_cache",
            self.vep_cache_path,
            "--force_overwrite",
            "--format",
            "vcf",
            "--vcf",
            "--no_stats",
            "--plugin",
            ','.join(['LoF',
                      'conservation_file:' + self.vep_conservation_path,
                      'loftee_path:' + self.vep_loftee_path,
                      'human_ancestor_fa:' + self.vep_human_ancestor_fa_path,
                      'gerp_bigwig:' + self.vep_gerp_bw_path]),
            "--dir_plugins",
            self.vep_loftee_path,
            "--custom",
            ",".join([self._vep_mis_db_path, "MisDb", 'vcf', "exact", "0", self.vep_mis_info_key]),
        ]

    @property
    def cmd_option_pick_one_gene_isoform(self) -> list:
        """Return options in order to pick a gene isoform
        with most severe consequence"""
        return [
            "--per_gene",
            "--pick",
            "--pick_order",
            "rank,canonical,appris,tsl,biotype,ccds,length",
        ]

    @property
    def cmd_option_pick_nearest_gene(self) -> list:
        """Return options in order to pick the nearest gene"""
        return ["--distance", "2000", "--nearest", "symbol", "--symbol"]
