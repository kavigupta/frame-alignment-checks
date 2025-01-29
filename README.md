
# Frame alignment checks

Set of tools for checking whether splicing prediction models are using frame alignment information. Does so on a set of
"long canonical internal coding exons" (see below for definition).

## Data sourcing

### Long canonical internal coding exons

These are computed using the following conditions.

 - appear in the SAM validation set (first half of the SpliceAI test set) of canonical exons in certain chromosomes
 - have exactly one ensembl annotation whose transcript is the same as the canonical transcript
 - start and end in a coding region
 - length at least 100nt

this set is built in to the package and does not need to be provided by the user.

The sources used to construct this are the SpliceAI dataset [1]
via the SAM repository's implementation (which is itself based on the SpliceAI implementation) [2]
and the Ensembl database [3].

### Relevant validation genes

These are sourced, like the long canonical internal coding exons, from the SAM validation set. Only genes relevant to
the long canonical internal coding exons are pulled.

### Minigenes

The minigenes are sourced from the hg19 canonical transcript, defined in the same way as SpliceAI's canonical transcript.

### Saturation Mutagenesis test benchmark

This is sourced from [https://genome.cshlp.org/content/suppl/2017/12/14/gr.219683.116.DC1/Supplemental_Table_S2.xlsx](this link) and is
cached in this package in case the link goes down.


[1]: Jaganathan, Kishore, et al. "Predicting splicing from primary sequence with deep learning." Cell 176.3 (2019): 535-548.
[2]: https://github.com/kavigupta/sam
[3]: https://useast.ensembl.org/index.html
