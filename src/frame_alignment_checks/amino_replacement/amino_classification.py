# From https://compbio.berkeley.edu/class/c246/Reading/dayhoff-1978-apss.pdf
# Figure 84 (p352)

amino_classification = {
    amino: amino_cat
    for amino_cat in ["C", "STPAG", "NDEQ", "HRK", "MILV", "FYW", "*"]
    for amino in amino_cat
}
