# tetrahedral_3D_cc_path.py -- step 3 of the tetrahedral tensor network: the contraction path
#
#     python tetrahedral_3D_cc_path.py structure_file path_file max_time
#
# Run by find_path_tetrahedral_3D_cc (tetrahedral_3D_cc_path_3.jl). The structure file has one
# line per tensor listing its labels; every label is a bit (dimension 2) shared by two or more
# tensors and summed over, so the network contracts to a single number. cotengra searches for
# max_time seconds for a good order of pairwise contractions and the best one is written to the
# path file, one pair per line:
#     a b       contract tensor a with tensor b
# The tensors are numbered 1, 2, ... as in the structure file, and the result of the k-th pair
# is tensor number (number of tensors + k). The lines starting with # give
#     W   the contraction width: the largest tensor met has 2^W entries (memory)
#     C   the contraction cost: the number of multiplications (time)

import sys
import cotengra as ctg

# (on Windows the parallel search starts new Python processes, which must not rerun the search)
if __name__ == "__main__":
    structure_file = sys.argv[1]
    path_file = sys.argv[2]
    max_time = float(sys.argv[3])

    # cotengra's fast path finders want every label as a single character, so label b
    # becomes the unicode character get_symbol(b)
    inputs = []
    for line in open(structure_file):
        inputs.append([ctg.get_symbol(int(b)) for b in line.split()])
    size_dict = {}
    for legs in inputs:
        for bond in legs:
            size_dict[bond] = 2

    methods = ["greedy", "labels", "random-greedy"]
    optimizer = ctg.HyperOptimizer(methods=methods, minimize="flops", max_time=max_time,
                                   max_repeats=10_000_000, reconf_opts={}, progbar=False)
    tree = optimizer.search(inputs, [], size_dict)

    W = tree.contraction_width()
    C = tree.contraction_cost()
    path = tree.get_ssa_path()

    out = open(path_file, "w")
    out.write(f"# tensors {len(inputs)}, labels {len(size_dict)}\n")
    out.write(f"# W = {W:.2f}   (largest tensor 2^W entries)\n")
    out.write(f"# C = {C:.4e}   (multiplications)\n")
    out.write(f"# cotengra {ctg.__version__}, {' + '.join(methods)}, minimize flops, {max_time:.0f} s, {len(optimizer.scores)} trials\n")
    for a, b in path:
        out.write(f"{a + 1} {b + 1}\n")
    out.close()

    print(f"{len(inputs)} tensors: W = {W:.2f}, C = {C:.4e}, {len(optimizer.scores)} trials -> {path_file}")
