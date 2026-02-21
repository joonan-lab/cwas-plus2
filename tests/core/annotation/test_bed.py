"""
Test cwas.core.annotation.bed — BED file annotation class.
"""
import pytest
from cwas.core.annotation.bed import annotate


def test_annotate_properties():
    inst = annotate(
        in_vcf_gz_path="/tmp/in.vcf.gz",
        out_vcf_path="/tmp/out.vcf.gz",
        annot_bed_path="/tmp/annot.bed.gz",
        num_proc=4,
    )
    assert inst.in_vcf_gz_path == "/tmp/in.vcf.gz"
    assert inst.out_vcf_path == "/tmp/out.vcf.gz"
    assert inst.annot_bed_path == "/tmp/annot.bed.gz"
    assert inst.num_proc == 4


def test_annotate_num_proc_cap():
    # Capped at 22 (one per autosomal chromosome)
    inst = annotate(
        in_vcf_gz_path="/tmp/in.vcf.gz",
        out_vcf_path="/tmp/out.vcf.gz",
        annot_bed_path="/tmp/annot.bed.gz",
        num_proc=30,
    )
    assert inst.num_proc == 22


def test_annotate_num_proc_at_boundary():
    inst = annotate(
        in_vcf_gz_path="/tmp/in.vcf.gz",
        out_vcf_path="/tmp/out.vcf.gz",
        annot_bed_path="/tmp/annot.bed.gz",
        num_proc=22,
    )
    assert inst.num_proc == 22


def test_annotate_num_proc_below_boundary():
    inst = annotate(
        in_vcf_gz_path="/tmp/in.vcf.gz",
        out_vcf_path="/tmp/out.vcf.gz",
        annot_bed_path="/tmp/annot.bed.gz",
        num_proc=1,
    )
    assert inst.num_proc == 1
