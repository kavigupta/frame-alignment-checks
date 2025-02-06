
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

### Phase handedness counts

This is a count of how many times each donor 9mer appears in each phase. This is sourced from the SpliceAI training set, via SAM.

### Non-stop donor windows

This is a collection of donors from the SpliceAI test set (again via SAM), specifically ones where swapping the donor 9mer
with an arbitrary sequence would not introduce a stop in the exon. Basically, we exclude conditions where the
flanking exon ends with a sequence that is a prefix of a stop codon, these are T, TA, and TG.

### Acceptor and donor LSSI models

These are models trained on the SpliceAI training set, via SAM. Copied directly from [https://github.com/kavigupta/sam/tree/main/spliceai/Canonical/splicepoint-models](here). We only use
these in tests, and they are not required for the package to run.

[1]: Jaganathan, Kishore, et al. "Predicting splicing from primary sequence with deep learning." Cell 176.3 (2019): 535-548.
[2]: https://github.com/kavigupta/sam
[3]: https://useast.ensembl.org/index.html
