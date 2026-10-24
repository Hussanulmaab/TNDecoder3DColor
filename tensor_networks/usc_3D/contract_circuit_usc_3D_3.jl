function contract_cubictn_usc_3D(tn::ContractionNetwork, max_bond_dim::Int, sc_bond_dim::Int; svd_eps::Float64=1e-16, do_gauging::Bool=false, canonicalness_threshold::Float64=1e-4, split_dim::Int=2, ignore_bond_tensors::Bool=false)

    xmin, xmax = extrema([k[1] for (k, v) in tn.tensors])
    ymin, ymax = extrema([k[2] for (k, v) in tn.tensors])
    zmin, zmax = extrema([k[3] for (k, v) in tn.tensors])

    # split up the bond tensors and put them onto sites
    if !ignore_bond_tensors
        for z = zmin:zmax
            for y = ymin:ymax
                for x = xmin:xmax
                    pos1 = (x, y, z)
                    for pos2 in neighbors(tn, pos1)
                        if pos2 > pos1
                            idx1 = commonind(tn[pos1], bond_matrix(tn, pos1, pos2))
                            idx2 = commonind(tn[pos2], bond_matrix(tn, pos1, pos2))
                            idxnew = Index(dim(idx1))
                            arr = sqrt.(Matrix(bond_matrix(tn, pos1, pos2), idx1, idx2))
                            tn[pos1] *= ITensor(arr, idx1, idxnew)
                            tn[pos2] *= ITensor(arr, idx2, idxnew)
                        end
                    end
                end
            end
        end
    end

    # set up PEPS 
    peps = generate_empty_PEPS(xmin, xmax, ymin, ymax)
    open_edge = Dict{Tuple{Int,Int},Index}()

    for y = ymin:ymax
        for x = xmin:xmax
            open_edge[x, y] = open_indices(peps, (x, y))[1]
            tn[(x, y, zmin)] *= ITensor([1.], open_edge[x, y])
        end
    end

    # contract
    expval = 0
    for z = zmin:zmax
        # contract two site bonds on same level
        for y = ymin:ymax
            for x = xmin:xmax
                for (x2, y2) in neighbors(peps, (x, y))
                    if (x2, y2) > (x, y)
                        downidx1 = commonind(peps[(x, y)], tn[(x, y, z)])
                        downidx2 = commonind(peps[(x2, y2)], tn[(x2, y2, z)])
                        link = commonind(tn[(x, y, z)], tn[(x2, y2, z)])
                        # println("(x, y): ", (x, y), "|downidx1", downidx1, "|downidx2", downidx2, "|link", link)
                        # println("peps[(x, y)]: \n", peps[(x, y)], "tn[(x, y, z)]: \n", tn[(x, y, z)])

                        U1, S1, V1 = svd(tn[(x, y, z)], [idx for idx in [downidx1, link] if !isnothing(idx)], maxdim=split_dim)
                        U2, S2, V2 = svd(tn[(x2, y2, z)], [idx for idx in [downidx2, link] if !isnothing(idx)], maxdim=split_dim)
                        #U1,S1,V1 = svd(tn[(x,y,z)], [idx for idx in [downidx1, link] if !isnothing(idx)], maxdim=max_bond_dim)
                        #U2,S2,V2 = svd(tn[(x2,y2,z)], [idx for idx in [downidx2, link] if !isnothing(idx)], maxdim=max_bond_dim)
                        upidx1 = commonind(U1, S1)
                        upidx2 = commonind(U2, S2)
                        tn[(x, y, z)] = S1 * V1
                        tn[(x2, y2, z)] = S2 * V2

                        expval += renormalize_tensor!(tn, (x, y, z))
                        expval += renormalize_tensor!(tn, (x2, y2, z))
                        h = Int(floor(log2(LinearAlgebra.norm(U1))))
                        U1 /= exp2(h)
                        expval += h
                        h = Int(floor(log2(LinearAlgebra.norm(U2))))
                        U2 /= exp2(h)
                        expval += h
                        expval += contract2_SU!(peps, (x, y), (x2, y2), U1 * U2, Vector{Index}([upidx1]), max_bond_dim, svd_eps)
                        expval += renormalize_tensor!(peps, (x, y))
                        expval += renormalize_tensor!(peps, (x2, y2))
                        expval += renormalize_bond_matrix!(peps, (x, y), (x2, y2))
                        open_edge[x, y] = upidx1
                        open_edge[x2, y2] = upidx2
                    end
                end
            end
        end

        # contract the final remaining R into the peps
        for x = xmin:xmax
            for y = ymin:ymax
                peps[(x, y)] *= tn[(x, y, z)]
                expval += renormalize_tensor!(peps, (x, y))
                if z < zmax
                    @assert length(open_indices(peps, (x, y))) == 1
                    open_edge[x, y] = open_indices(peps, (x, y))[1]
                end
            end
        end

        # gauge
        if z < zmax && do_gauging
            n_steps = 0
            while canonicalness(peps) > canonicalness_threshold && n_steps < 100
                for x = xmin:xmax
                    for y = ymin:ymax
                        for (x2, y2) in neighbors(peps, (x, y))
                            if (x2, y2) > (x, y)
                                expval += identity_SU!(peps, (x, y), (x2, y2), max_bond_dim, svd_eps)
                                expval += renormalize_tensor!(peps, (x, y))
                                expval += renormalize_tensor!(peps, (x2, y2))
                                expval += renormalize_bond_matrix!(peps, (x, y), (x2, y2))
                            end
                        end
                    end
                end
            end
        end
    end

    # contract final PEPS with sweep contractor
    res = contract_PEPS(peps, sc_bond_dim, svd_eps)
    return (res[1], res[2] + expval)
end