
# Frame alignment checks

Set of tools for checking whether splicing prediction models are using frame alignment information. Does so on a set of
"long canonical internal coding exons", that is, exons that

 - appear in the SAM validation set (first half of the SpliceAI test set) of canonical exons in certain chromosomes
 - have exactly one ensembl annotation whose transcript is the same as the canonical transcript
 - start and end in a coding region
 - length at least 100nt

this set is built in to the package and does not need to be provided by the user.