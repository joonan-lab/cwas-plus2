import argparse
import shutil
from pathlib import Path
import os

import cwas.core.configuration.create as create
import cwas.utils.log as log
import cwas.utils.check as check
from cwas.runnable import Runnable


class Configuration(Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.workspace = self._get_workspace()
        self.user_config = self.workspace / "configuration.txt"
        self.data_dir_symlink = self.workspace / "annotation-data"
        self.gene_matrix_symlink = self.workspace / "gene_matrix.txt"
        self.bed_key_list_symlink = self.workspace / "annotation_keys.yaml"
        self.category_domain_list = self.workspace / "category_domain.yaml"
        self.redundant_category_table = (
            self.workspace / "redundant_category.txt"
        )

    def _get_workspace(self):
        workspace = self.get_env("CWAS_WORKSPACE")
        if workspace is None:
            raise RuntimeError(
                "${CWAS_WORKSPACE} does not set. Run the Start step first."
            )
        return Path(workspace)

    @staticmethod
    def _print_args(args: argparse.Namespace):
        log.print_arg(
            "Force to overwrite configuration",
            "Y" if args.force_overwrite else "N",
        )

    @property
    def force_overwrite(self):
        return bool(self.args.force_overwrite)

    def run(self):
        self._set_config_to_attr()
        self._check_attr_from_user_config()
        self._create_data_dir_symlink()
        self._create_gene_matrix_symlink()
        self._create_bed_key_list_symlink()
        self._create_category_info()
        self._set_env()

    def _set_config_to_attr(self):
        user_config = self._load_configuration()
        self.data_dir = Path(user_config.get("ANNOTATION_DATA_DIR"))
        self.gene_matrix = self.data_dir.joinpath(user_config.get("GENE_MATRIX"))
        self.vep = Path(user_config.get("VEP"))
        self.vep_cache_dir = Path(user_config.get("VEP_CACHE_DIR"))
        self.vep_conservation = self.vep_cache_dir.joinpath(user_config.get("VEP_CONSERVATION_FILE"))
        self.vep_loftee = self.vep_cache_dir.joinpath(user_config.get("VEP_LOFTEE"))
        self.vep_human_ancestor_fa = self.vep_cache_dir.joinpath(user_config.get("VEP_HUMAN_ANCESTOR_FA"))
        self.vep_gerp_bw = self.vep_cache_dir.joinpath(user_config.get("VEP_GERP_BIGWIG"))
        self.vep_mis_db = self.vep_cache_dir.joinpath(user_config.get("VEP_MIS_DB"))
        self.vep_mis_info_key = user_config.get("VEP_MIS_INFO_KEY")
        self.vep_mis_thres = Path(user_config.get("VEP_MIS_THRES"))

        annot_key_conf = user_config.get("ANNOTATION_KEY_CONFIG")
        self.annot_key_conf = (
            None if not annot_key_conf else self.data_dir.joinpath(annot_key_conf)
        )

    def _load_configuration(self) -> dict:
        user_config_dict = {}

        with self.user_config.open() as user_config_file:
            for line in user_config_file:
                k, v = line.strip().split("=")
                # The two 'CONFIG' file paths are optional.
                if not k.endswith("CONFIG"):
                    self._check_config_value(k, v)
                user_config_dict[k] = v

        return user_config_dict

    def _check_config_value(self, key: str, value: str):
        if not value:
            raise ValueError(f"The value for '{key}' is empty.")

    def _check_attr_from_user_config(self):
        check.check_is_file(self.user_config)
        check.check_is_dir(self.data_dir)
        check.check_is_dir(self.vep_cache_dir)        
        check.check_is_file(self.vep_conservation)
        check.check_is_dir(self.vep_loftee)
        check.check_is_file(self.vep_human_ancestor_fa)
        check.check_is_file(self.vep_gerp_bw)
        check.check_is_file(self.vep_mis_db)
        check.check_is_file(self.gene_matrix)
        if self.annot_key_conf is not None:
            check.check_is_file(self.annot_key_conf)
        

        if self.vep is not None and shutil.which(self.vep) is None:
            raise ValueError(f'"{self.vep}" is not an executable.')

    def _create_symlink(self, target, link_path, is_directory=False):
        log.print_progress(f'Create a symlink "{link_path}" -> "{target}"')
        try:
            link_path.symlink_to(target, target_is_directory=is_directory)
        except FileExistsError:
            if self.force_overwrite:
                log.print_warn(
                    f'"{link_path}" already exists, removing it and creating the symbolic link again.'
                )
                temp_link = Path(str(link_path) + ".new")
                if is_directory:
                    os.unlink(link_path)
                else:
                    os.remove(link_path)
                os.symlink(target, temp_link)
                os.rename(temp_link, link_path)
            else:
                log.print_warn(
                    f'"{link_path}" already exists so skip '
                    f"creating the symbolic link"
                )

    def _create_data_dir_symlink(self):
        self._create_symlink(self.data_dir, self.data_dir_symlink, is_directory=True)

    def _create_gene_matrix_symlink(self):
        self._create_symlink(self.gene_matrix, self.gene_matrix_symlink)

    def _create_bed_key_list_symlink(self):
        self._create_symlink(self.annot_key_conf, self.bed_key_list_symlink)

    def _create_category_info(self):
        """ Create a list of category domains and a redundant category table"""
        domain_list = getattr(self, "category_domain_list")
        annot_key_conf = getattr(self, "annot_key_conf")
        gene_matrix = getattr(self, "gene_matrix")
        redundant_category_table = getattr(self, "redundant_category_table")

        log.print_progress(
            f"Create a CWAS category domain list " f'"{domain_list}"'
        )
        create.create_category_domain_list(
            domain_list, annot_key_conf, gene_matrix
        )
        log.print_progress(
            f"Create a redundant category table "
            f'"{redundant_category_table}"'
        )
        create.create_redundant_category_table(redundant_category_table)

    def _set_env(self):
        log.print_progress("Set CWAS environment variables")
        self.set_env("VEP", getattr(self, "vep"))
        self.set_env("VEP_CACHE_DIR", getattr(self, "vep_cache_dir"))
        self.set_env("VEP_CONSERVATION_FILE", getattr(self, "vep_conservation"))
        self.set_env("VEP_LOFTEE", getattr(self, "vep_loftee"))
        self.set_env("VEP_HUMAN_ANCESTOR_FA", getattr(self, "vep_human_ancestor_fa"))
        self.set_env("VEP_GERP_BIGWIG", getattr(self, "vep_gerp_bw"))
        self.set_env("VEP_MIS_DB", getattr(self, "vep_mis_db"))
        self.set_env("VEP_MIS_INFO_KEY", getattr(self, "vep_mis_info_key"))
        self.set_env("VEP_MIS_THRES", getattr(self, "vep_mis_thres"))
        self.set_env("ANNOTATION_DATA", self.data_dir_symlink)
        self.set_env("GENE_MATRIX", self.gene_matrix_symlink)
        self.set_env("ANNOTATION_BED_KEY", self.bed_key_list_symlink)
        self.set_env("CATEGORY_DOMAIN", self.category_domain_list)
        self.set_env("REDUNDANT_CATEGORY", self.redundant_category_table)
        self.save_env()
