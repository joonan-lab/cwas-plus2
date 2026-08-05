from pathlib import Path
from typing import Optional
import argparse
import gzip
import re
import zarr
from cwas.core.categorization.parser import _parse_annot_field
import numpy as np
import pandas as pd
from cwas.runnable import Runnable
from cwas.utils.log import print_progress, print_warn
from cwas.core.categorization.parser import (
    parse_annotated_vcf,
)
from cwas.core.common import (
    GENE_ID_SEPARATOR,
    GENE_ID_SUFFIX_PATTERN,
    check_gene_matrix_ids,
    check_gene_list_values,
    find_gene_key_columns,
    is_gene_list_member,
    resolve_tied_gene_id,
)
from cwas.utils.check import check_is_file, check_is_dir
from numcodecs import JSON

class ExtractVariant(Runnable):
    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)
        self._annotated_vcf = None
        self._gene_matrix = None
        self._gene_symbols = None
        self._tag = None
        self._category_set_path = None
        self._category_set = None
        self._annot_field_names = None

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_is_file(args.input_path)
        check_is_dir(args.output_dir_path)
        if args.category_set_path :
            check_is_file(args.category_set_path)

    @property
    def input_path(self):
        return self.args.input_path.resolve()

    @property
    def output_dir_path(self):
        return self.args.output_dir_path.resolve()

    @property
    def annotation_info(self) -> bool:
        return self.args.annotation_info

    @property
    def tag(self) -> str:
        return self.args.tag

    @property
    def mis_info_key(self) -> str:
        return self.get_env("VEP_MIS_INFO_KEY")

    @property
    def mis_thres(self) -> float:
        return float(self.get_env("VEP_MIS_THRES"))

    @property
    def annotated_vcf(self) -> pd.DataFrame:
        if self._annotated_vcf is None:
            print_progress("Parse the annotated VCF")
            self._annotated_vcf = parse_annotated_vcf(
                Path(self.input_path)
            )
        return self._annotated_vcf
    
    @property
    def annot_field_names(self) -> list:
        if self._annot_field_names is None:
            with gzip.open(self.input_path, "rb") as vep_vcf_file:
                for line_bytes in vep_vcf_file:
                    line = line_bytes.decode("utf-8")
                    if line.startswith("#"):
                        if line.startswith("##INFO=<ID=ANNOT"):
                            self._annot_field_names = _parse_annot_field(line)
        return self._annot_field_names

    @property
    def gene_matrix(self) -> pd.DataFrame:
        if self._gene_matrix is None:
            # The gene list values are read as text, exactly as the
            # categorization step reads them, so that the two steps cannot
            # disagree on which genes a gene list holds. They are cast back to
            # integers once checked, because the annotation that follows counts
            # on them being numbers.
            gene_matrix = pd.read_csv(
                self.get_env("GENE_MATRIX"),
                sep='\t',
                dtype=str,
                encoding="utf-8-sig",
            )
            # The gene key columns are found by name, the same way the
            # categorization step finds them, so that a gene matrix accepted
            # there is accepted here too.
            columns = gene_matrix.columns.tolist()
            gene_id_idx, gene_name_idx = find_gene_key_columns(columns)
            gene_name_col = (
                columns[gene_name_idx] if gene_name_idx is not None else None
            )

            gene_matrix = gene_matrix.rename(
                columns={columns[gene_id_idx]: 'gene_id'}
            )

            check_gene_matrix_ids(
                gene_matrix['gene_id'].tolist(),
                self.get_env("GENE_MATRIX"),
                gene_matrix.index.to_numpy() + 2,
            )

            gene_list_cols = [
                col
                for col in gene_matrix.columns
                if col not in ('gene_id', gene_name_col)
            ]
            for col in gene_list_cols:
                check_gene_list_values(gene_matrix[col].unique())
            gene_matrix[gene_list_cols] = gene_matrix[gene_list_cols].astype(
                int
            )
            gene_matrix['gene_id'] = (
                gene_matrix['gene_id']
                .astype(str)
                .str.replace(GENE_ID_SUFFIX_PATTERN, "", regex=True)
            )

            # Stripping the version and '_PAR_Y' suffixes can collapse two rows
            # onto the same gene ID. A duplicated key would multiply variant
            # rows in the merge of annotate_variants, which then desynchronizes
            # the annotation block appended positionally afterwards, so the
            # matrix is de-duplicated here.
            duplicated = gene_matrix['gene_id'].duplicated(keep='last')
            if duplicated.any():
                self._warn_conflicting_duplicates(gene_matrix, gene_name_col)
                gene_matrix = gene_matrix[~duplicated]

            # Genes are matched by ID, so the symbol is kept apart and only
            # used to label the output.
            self._gene_symbols = (
                gene_matrix.set_index('gene_id')[gene_name_col]
                if gene_name_col is not None
                else None
            )
            self._gene_matrix = gene_matrix.drop(
                columns=[gene_name_col] if gene_name_col is not None else []
            )
        return self._gene_matrix

    @property
    def category_set_path(self) -> Optional[Path]:
        return (
            self.args.category_set_path.resolve()
            if self.args.category_set_path
            else None
        )
    @property
    def category_set(self) -> pd.DataFrame:
        if self._category_set is None and self.category_set_path:
            self._category_set = pd.read_csv(self.category_set_path, sep='\t')
        return self._category_set
    
    @property
    def result_path(self) -> Path:
        if self.tag is None:
            save_name = 'extracted_variants.txt.gz'
        else:
            save_name = '.'.join([self.tag, 'extracted_variants.txt.gz'])
        f_name = re.sub(r'annotated\.vcf\.gz|annotated\.vcf', save_name, self.input_path.name)
        return self.output_dir_path / f_name
    
    def extract_idx_by_int(self, n: int) -> list:
        """Get an index from the input list by using the input integer"""
        i = 0
        result = []

        while n != 0:
            if n % 2 == 1:
                result.append(i)
            n >>= 1
            i += 1

        return result

    @staticmethod
    def _warn_conflicting_duplicates(
        gene_matrix: pd.DataFrame, gene_name_col: Optional[str]
    ) -> None:
        """ Warn about gene IDs whose duplicated rows carry different gene lists.

        Rows collapse onto one ID when a gene matrix carries several versions of
        a gene, or the two pseudoautosomal copies of one. That is expected and
        silent. Only rows that disagree on their gene lists are worth a warning,
        because there the entry that is dropped carried something the kept one
        does not.
        """
        gene_list_cols = [
            col
            for col in gene_matrix.columns
            if col not in ('gene_id', gene_name_col)
        ]
        dup_rows = gene_matrix[
            gene_matrix['gene_id'].duplicated(keep=False)
        ]
        conflicting = (
            dup_rows.groupby('gene_id')[gene_list_cols]
            .nunique()
            .max(axis=1)
            .gt(1)
        )

        if conflicting.any():
            print_warn(
                f"{int(conflicting.sum())} gene ID(s) are duplicated in the "
                "gene matrix with differing gene lists. Only the last entry "
                "of each is kept."
            )

    def _tied_gene_sets(self, tied_gene_ids: pd.Series) -> dict:
        """ Map each gene of a tie to the gene lists it belongs to.

        Only the genes that actually appear in a tie are looked up, because
        ties are rare and the gene matrix holds tens of thousands of genes.
        """
        candidates = {
            gene
            for value in tied_gene_ids.unique()
            for gene in value.split(GENE_ID_SEPARATOR)
        }
        gene_matrix = self.gene_matrix
        gene_list_cols = [
            col for col in gene_matrix.columns if col != 'gene_id'
        ]
        rows = gene_matrix[gene_matrix['gene_id'].isin(candidates)]
        return {
            row['gene_id']: {
                col for col in gene_list_cols if is_gene_list_member(row[col])
            }
            for _, row in rows.iterrows()
        }

    def annotate_variants(self):
        print_progress("Annotate variants with annotation dataset")
        self.annotated_vcf['CLASS'] = np.where(self.annotated_vcf['REF'].str.len() > 1, 'Deletion',
                                               np.where((self.annotated_vcf['REF'].str.len() == 1) & (self.annotated_vcf['ALT'].str.len() == 1), 'SNV', 'Insertion'))
        is_intergenic = (
            self.annotated_vcf['Consequence'].str.contains("downstream_gene_variant")
            | self.annotated_vcf['Consequence'].str.contains("intergenic_variant")
        )
        self.annotated_vcf['DEF.GENE_ID'] = (
            np.where(is_intergenic,
                     self.annotated_vcf['NEAREST'],
                     self.annotated_vcf['Gene'])
        )
        self.annotated_vcf['DEF.GENE_ID'] = (
            pd.Series(self.annotated_vcf['DEF.GENE_ID'], index=self.annotated_vcf.index)
            .astype(str)
            .str.replace(GENE_ID_SUFFIX_PATTERN, "", regex=True)
        )
        # VEP joins the genes tied for nearest with '&'. They are resolved to
        # a single gene by the same rule categorization uses, so that the
        # extracted variants agree with the categories they were counted in.
        tied = self.annotated_vcf['DEF.GENE_ID'].str.contains(
            GENE_ID_SEPARATOR, regex=False
        )
        if tied.any():
            tied_gene_sets = self._tied_gene_sets(
                self.annotated_vcf.loc[tied, 'DEF.GENE_ID']
            )
            self.annotated_vcf.loc[tied, 'DEF.GENE_ID'] = (
                self.annotated_vcf.loc[tied, 'DEF.GENE_ID'].map(
                    lambda gene: resolve_tied_gene_id(gene, tied_gene_sets)
                )
            )
        merged_df = pd.merge(self.annotated_vcf, self.gene_matrix, left_on='DEF.GENE_ID', right_on='gene_id', how='left')
        cols_to_fillna = self.gene_matrix.columns.tolist()
        merged_df[cols_to_fillna] = merged_df[cols_to_fillna].fillna(0)
        # Report the gene symbol for readability, falling back to the gene ID
        # for genes that are absent from the gene matrix.
        merged_df['DEF.GENE'] = (
            merged_df['DEF.GENE_ID'].map(self._gene_symbols).fillna(merged_df['DEF.GENE_ID'])
            if self._gene_symbols is not None
            else merged_df['DEF.GENE_ID']
        )
        merged_df = merged_df.drop(columns=['gene_id'])
        ## Coding
        # Define the list of string patterns to search for
        patterns = ['stop_gained', 'splice_donor', 'splice_acceptor', 'frameshift_variant', 'missense_variant', 'protein_altering_variant', 'start_lost', 'stop_lost', 'inframe_deletion', 'inframe_insertion', 'synonymous_variant', 'stop_retained_variant', 'incomplete_terminal_codon_variant', 'protein_altering_variant', 'coding_sequence_variant']
        # Use str.contains() to search for each pattern in the Consequence column
        merged_df['is_CodingRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_CodingRegion'] = np.where(merged_df['ProteinCoding'] == 1, merged_df['is_CodingRegion'], 0)
        ## Noncoding
        merged_df['is_NoncodingRegion'] = np.where(merged_df['is_CodingRegion'] == 1, 0, 1)
        ## PTV
        # Define the list of string patterns to search for
        patterns = ['stop_gained', 'splice_donor', 'splice_acceptor', 'frameshift_variant']
        # Use str.contains() to search for each pattern in the Consequence column
        merged_df['is_PTVRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_PTVRegion'] = np.where((merged_df['is_CodingRegion'] == 1) & (merged_df['LoF'] == 'HC') & ((merged_df['LoF_flags'] == 'SINGLE_EXON') | (merged_df['LoF_flags'] == '')), merged_df['is_PTVRegion'], 0)
        ## Frameshift
        # Define the list of string patterns to search for
        patterns = ['frameshift_variant']
        # Define the list of string patterns to exclude
        exclude_patterns = ['stop_gained', 'splice_donor', 'splice_acceptor']
        merged_df['is_FrameshiftRegion'] = ((merged_df['Consequence'].str.contains('|'.join(patterns))) 
                                            & (~merged_df['Consequence'].str.contains('|'.join(exclude_patterns)))).astype(int)
        merged_df['is_FrameshiftRegion'] = np.where(merged_df['is_PTVRegion'] == 1, merged_df['is_FrameshiftRegion'], 0)
        ## Missense
        patterns = ['missense_variant', 'protein_altering_variant', 'start_lost', 'stop_lost']
        merged_df['is_MissenseRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_MissenseRegion'] = np.where((merged_df['is_CodingRegion'] == 1) & (merged_df['is_PTVRegion'] == 0), merged_df['is_MissenseRegion'], 0)
        ## Damaging missense
        merged_df['is_DamagingMissenseRegion'] = np.where((merged_df['is_MissenseRegion'] == 1) & (pd.to_numeric(merged_df["MisDb_" + self.mis_info_key], errors='coerce').fillna(0) >= self.mis_thres), 1, 0)
        # Define the list of string patterns to search for
        patterns = ['inframe_deletion', 'inframe_insertion']
        merged_df['is_InFrameRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_InFrameRegion'] = np.where((merged_df['is_CodingRegion'] == 1) & (merged_df['is_PTVRegion'] == 0) & (merged_df['is_MissenseRegion'] == 0), merged_df['is_InFrameRegion'], 0)
        ## Silent
        patterns = ['synonymous_variant']
        merged_df['is_SilentRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_SilentRegion'] = np.where((merged_df['is_CodingRegion'] == 1) &
                                                (merged_df['is_PTVRegion'] == 0) &
                                                (merged_df['is_MissenseRegion'] == 0) &
                                                (merged_df['is_InFrameRegion'] == 0),
                                                merged_df['is_SilentRegion'], 0)
        ## UTRs
        # Define the list of string patterns to search for
        patterns = ['3_prime_UTR_variant']
        merged_df['is_3PrimeUTRsRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_3PrimeUTRsRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1),
                                                    merged_df['is_3PrimeUTRsRegion'], 0)
        patterns = ['5_prime_UTR_variant']
        merged_df['is_5PrimeUTRsRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_5PrimeUTRsRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1),
                                                    merged_df['is_5PrimeUTRsRegion'], 0)

        merged_df['is_UTRsRegion'] = np.where((merged_df['is_3PrimeUTRsRegion'] == 1) | (merged_df['is_5PrimeUTRsRegion'] == 1),
                                              1, 0)          
        
        ## Promoter
        patterns = ['upstream_gene_variant']
        merged_df['is_PromoterRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_PromoterRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0),
                                                  merged_df['is_PromoterRegion'], 0)

        ## SpliceSite non-canonical
        patterns = ['splice_region_variant']
        merged_df['is_SpliceSiteRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_SpliceSiteRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0) &
                                                  (merged_df['is_PromoterRegion'] == 0),
                                                  merged_df['is_SpliceSiteRegion'], 0)

        ## Intron
        patterns = ['intron_variant']
        merged_df['is_IntronRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_IntronRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0) &
                                                  (merged_df['is_PromoterRegion'] == 0) &
                                                  (merged_df['is_SpliceSiteRegion'] == 0),
                                                  merged_df['is_IntronRegion'], 0)

        ## Intergenic
        patterns = ['downstream_gene_variant', 'intergenic_variant']
        merged_df['is_IntergenicRegion'] = merged_df['Consequence'].str.contains('|'.join(patterns)).astype(int)
        merged_df['is_IntergenicRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0) &
                                                  (merged_df['is_PromoterRegion'] == 0) &
                                                  (merged_df['is_IntronRegion'] == 0) &
                                                  (merged_df['is_SpliceSiteRegion'] == 0),
                                                  merged_df['is_IntergenicRegion'], 0)
        ## lincRNA
        merged_df['is_lincRnaRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0) &
                                                  (merged_df['is_PromoterRegion'] == 0) &
                                                  (merged_df['is_IntronRegion'] == 0) &
                                                  (merged_df['is_SpliceSiteRegion'] == 0) &
                                                  (merged_df['is_IntergenicRegion'] == 0) &
                                                  (merged_df['lincRNA'] == 1) &
                                                  (merged_df['ProteinCoding'] == 0),
                                                  1, 0)
        ## Others
        merged_df['is_OtherTranscriptRegion'] = np.where((merged_df['is_NoncodingRegion'] == 1) &
                                                  (merged_df['is_UTRsRegion'] == 0) &
                                                  (merged_df['is_PromoterRegion'] == 0) &
                                                  (merged_df['is_IntronRegion'] == 0) &
                                                  (merged_df['is_SpliceSiteRegion'] == 0) &
                                                  (merged_df['is_IntergenicRegion'] == 0) &
                                                  (merged_df['is_lincRnaRegion'] == 0) &
                                                  (merged_df['ProteinCoding'] == 0),
                                                  1, 0)
        
        root_data = np.zeros((len(self.annotated_vcf), len(self.annot_field_names)))
        for i in range(0, self.annotated_vcf.shape[0]):
            index = self.extract_idx_by_int(int(self.annotated_vcf.loc[i,'ANNOT']))
            if len(index)!=0:
                root_data[i, index] = 1
        root_df = pd.DataFrame(root_data,
                               columns=self.annot_field_names)
        # Append df2 to df1 vertically (row-wise)
        self._result = pd.concat([merged_df, root_df], axis=1, ignore_index=False)

    def allocate_variants(self, category: pd.DataFrame):
        if category['variant_type'] == 'All':
            filtered_result = self._result[self._result['CLASS'].isin(['SNV', 'Deletion', 'Insertion'])]
        elif category['variant_type'] == 'SNV':
            filtered_result = self._result.query('CLASS == "SNV"')
        else:
            filtered_result = self._result[self._result['CLASS'].isin(['Deletion', 'Insertion'])]

        if category['gene_set'] != 'Any':
            filtered_result = filtered_result[filtered_result[category['gene_set']] == 1]
        if category['functional_score'] != 'All':
            filtered_result = filtered_result[filtered_result[category['functional_score']] == 1]
        if category['gencode'] != 'Any':
            filtered_result = filtered_result[filtered_result['_'.join(['is', category['gencode']])] == 1]
        if category['functional_annotation'] != 'Any':
            filtered_result = filtered_result[filtered_result[category['functional_annotation']] == 1]
        filtered_result['CATEGORY'] = category['Category']
    
        return(filtered_result)

    def filter_variants(self):
        print_progress(f"Filter variants in {self.category_set.shape[0]} categories")
        self.category_set[['variant_type', 'gene_set', 'functional_score', 'gencode', 'functional_annotation']] = self.category_set['Category'].str.split('_', expand=True)
        # Filter variants by categories and concatenate them vertically
        self._result = pd.concat(self.category_set.apply(lambda x: self.allocate_variants(category = x), axis=1).tolist(), axis=0)
    
    def remove_annotation_info(self):
        print_progress("No annotation information attached")
        if self.category_set_path :
            self._result = self._result.loc[:, ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'SAMPLE', 'CATEGORY']]
        else :
            self._result = self._result.loc[:, ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'SAMPLE']]
    
    def save_result(self):
        print_progress(f"Save the result to the file {self.result_path}")
        self._result.to_csv(self.result_path, sep='\t', compression='gzip', index=False)
        #root = zarr.open(self.result_path, mode = 'w')
        #root.create_group('metadata')
        #root['metadata'].attrs['columns'] = self._result.columns.tolist()
        #root.create_dataset('data', data = self._result.values, chunks=(1000, 1000), dtype=object, object_codec=JSON())
    
    def run(self):
        self.annotate_variants()
        if self.category_set_path :
            self.filter_variants()
        if self.annotation_info is None :
            self.remove_annotation_info()
        else:
            print_progress("Annotation information attached")
        self.save_result()
        print_progress("Done")
