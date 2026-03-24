.. _annotation:

*********************************
Annotation
*********************************

CWAS-Plus2 annotates variants using VEP and its own custom annotation algorithm. The parameters of the command are as below:

- -v, --vcf_file: Path to the input vcf file. This file could be bgzipped or not.
- -p, --num_proc: Number of worker processes that will be used for the annotation process. By default, 1.
- -o_dir, --output_directory: Path to the directory where the output files will be saved. By default, outputs will be saved at ``$CWAS_WORKSPACE``.
- --docker-mode: Run VEP using Docker (``ensemblorg/ensembl-vep``) instead of a local binary. With this option, a local VEP installation is not required. By default, False.
- --vep-version: VEP version for Docker image tag (e.g. ``115.0``). Only used with ``--docker-mode``. If omitted, uses ``latest``.


.. code-block:: solidity

  cwas annotation -v INPUT.vcf -o_dir OUTPUT_DIR -p 8

To run with Docker instead of a local VEP installation:

.. code-block:: solidity

  cwas annotation -v INPUT.vcf -o_dir OUTPUT_DIR -p 8 --docker-mode --vep-version 115.0