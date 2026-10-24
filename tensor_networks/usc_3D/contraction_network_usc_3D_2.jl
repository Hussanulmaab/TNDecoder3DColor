const Pos3D = Tuple{Int,Int,Int}

struct CubicTNTensorDescription
    type::TensorType
    val::Tuple{Float64,Float64}
end

struct CubicTN
    xmin::Int
    xmax::Int
    ymin::Int
    ymax::Int
    zmin::Int
    zmax::Int

    sites::Dict{Pos3D,CubicTNTensorDescription}
    one_site_bonds::Dict{Pos3D,CubicTNTensorDescription}
    two_site_bonds::Dict{Set{Pos3D},CubicTNTensorDescription}
end

struct ContractionNetwork{P}
    tensors::Dict{P,ITensor}
    adj::Dict{P,Vector{P}}
    bond_matrices::Dict{Set{P},ITensor}
end

function Base.getindex(net::ContractionNetwork{P}, site::P) where P
    return net.tensors[site]
end

function Base.setindex!(net::ContractionNetwork{P}, T::ITensor, site::P) where P
    net.tensors[site] = T
end

function neighbors(net::ContractionNetwork{P}, site::P) where {P}
    return net.adj[site]
end

function open_indices(net::ContractionNetwork{P}, site::P)::Vector{Index} where P
    return setdiff(inds(net.tensors[site]), [inds(bond_matrix(net, site, site2)) for site2 in neighbors(net, site)]...)
end

function bond_matrix(net::ContractionNetwork{P}, site1::P, site2::P) where P
    return net.bond_matrices[Set([site1, site2])]
end

function generate_empty_PEPS(xmin::Int, xmax::Int, ymin::Int, ymax::Int)
    tensors = Dict{Tuple{Int,Int},ITensor}()
    adj = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    bond_matrices = Dict{Set{Tuple{Int,Int}},ITensor}()
    neighbors(x, y) = [(x2, y2) for (x2, y2) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if (xmin <= x2 <= xmax && ymin <= y2 <= ymax)]


    bonds = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Index}()
    for y = ymin:ymax
        for x = xmin:xmax
            adj[x, y] = neighbors(x, y)
            for (x2, y2) in adj[x, y]
                if (x2, y2) > (x, y)
                    idx1 = Index(1)
                    idx2 = Index(1)
                    bonds[(x, y), (x2, y2)] = idx1
                    bonds[(x2, y2), (x, y)] = idx2
                    bond_matrices[Set([(x, y), (x2, y2)])] = ITensor([1.], idx1, idx2)
                end
            end
        end
    end

    for y = ymin:ymax
        for x = xmin:xmax
            tensors[x, y] = ITensor([1.], [bonds[(x, y), (x2, y2)] for (x2, y2) in adj[x, y]]..., Index(1))
        end
    end

    return ContractionNetwork{Tuple{Int,Int}}(tensors, adj, bond_matrices)
end

function generate_empty_cubictn(xmin::Int, xmax::Int, ymin::Int, ymax::Int, zmin::Int, zmax::Int)
    tensors = Dict{Tuple{Int,Int,Int},ITensor}()
    adj = Dict{Tuple{Int,Int,Int},Vector{Tuple{Int,Int,Int}}}()
    bond_matrices = Dict{Set{Tuple{Int,Int,Int}},ITensor}()
    neighbors(x, y, z) = [(x2, y2, z2) for (x2, y2, z2) in [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)] if (xmin <= x2 <= xmax && ymin <= y2 <= ymax && zmin <= z2 <= zmax)]

    bonds = Dict{Tuple{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},Index}()
    for z = zmin:zmax
        for y = ymin:ymax
            for x = xmin:xmax
                adj[x, y, z] = neighbors(x, y, z)
                for (x2, y2, z2) in adj[x, y, z]
                    if (x2, y2, z2) > (x, y, z)
                        idx1 = Index(1)
                        idx2 = Index(1)
                        bonds[(x, y, z), (x2, y2, z2)] = idx1
                        bonds[(x2, y2, z2), (x, y, z)] = idx2
                        bond_matrices[Set([(x, y, z), (x2, y2, z2)])] = ITensor([1.], idx1, idx2)
                    end
                end
            end
        end
    end
    for z = zmin:zmax
        for y = ymin:ymax
            for x = xmin:xmax
                tensors[x, y, z] = ITensor([1.], [bonds[(x, y, z), (x2, y2, z2)] for (x2, y2, z2) in adj[x, y, z]]..., Index(1))
            end
        end
    end

    return ContractionNetwork{Tuple{Int,Int,Int}}(tensors, adj, bond_matrices)
end

function contract2_SU!(net::ContractionNetwork{P}, site1::P, site2::P, T::ITensor, indices1_out::Vector{Index}, max_bond_dim::Int, svd_eps::Float64=1e-16) where P
    @assert length(commoninds(net[site1], T)) > 0
    @assert length(commoninds(net[site2], T)) > 0
    @assert site1 in neighbors(net, site2) && site2 in neighbors(net, site1)

    # contract environment tensors
    T1 = net[site1]
    T2 = net[site2]
    for site3 in neighbors(net, site1)
        if site3 != site2
            T1 *= bond_matrix(net, site1, site3)
        end
    end
    for site3 in neighbors(net, site2)
        if site3 != site1
            T2 *= bond_matrix(net, site2, site3)
        end
    end
    expval = 0
    h = Int(floor(log2(LinearAlgebra.norm(T1))))
    T1 /= exp2(h)
    expval += h
    h = Int(floor(log2(LinearAlgebra.norm(T2))))
    T2 /= exp2(h)
    expval += h

    # QR decomposition
    Q1, R1 = qr(T1, [commonind(T1, net[site3]) for site3 in neighbors(net, site1) if site3 != site2]...)
    Q2, R2 = qr(T2, [commonind(T2, net[site3]) for site3 in neighbors(net, site2) if site3 != site1]...)

    # contract gate
    theta = R1 * bond_matrix(net, site1, site2) * R2 * T

    # SVD and truncate
    U, S, V = svd(theta, commonind(Q1, R1), open_indices(net, site1)..., indices1_out..., maxdim=max_bond_dim, cutoff=svd_eps)

    # extract environment again 
    U = U * Q1
    V = V * Q2
    for site3 in neighbors(net, site1)
        if site3 != site2
            bondmat_inds = inds(bond_matrix(net, site1, site3))
            bondmat_inv = ITensor(diag_inv(Array(bond_matrix(net, site1, site3), bondmat_inds...)), bondmat_inds...)
            U *= bondmat_inv
        end
    end
    for site3 in neighbors(net, site2)
        if site3 != site1
            bondmat_inds = inds(bond_matrix(net, site2, site3))
            bondmat_inv = ITensor(diag_inv(Array(bond_matrix(net, site2, site3), bondmat_inds...)), bondmat_inds...)
            V *= bondmat_inv
        end
    end

    # store updated tensors
    net[site1] = U
    net[site2] = V
    net.bond_matrices[Set([site1, site2])] = S

    return expval
end

function contract_PEPS(peps::ContractionNetwork{P}, sc_bond_dim::Int, svd_eps::Float64=1e-16) where P
    xmin, xmax = extrema([pos[1] for (pos, v) in peps.tensors])
    ymin, ymax = extrema([pos[2] for (pos, v) in peps.tensors])

    # contract bond matrices 
    for x = xmin:xmax
        for y = ymin:ymax
            for (x2, y2) in neighbors(peps, (x, y))
                if (x2, y2) > (x, y)
                    peps.tensors[x, y] *= bond_matrix(peps, (x, y), (x2, y2))
                end
            end
        end
    end

    # we sweep from ymin to ymax
    expval = 0
    mps = MPS([peps.tensors[x, ymin] for x = xmin:xmax])
    for y = (ymin+1):ymax
        # contract
        for x = xmin:xmax
            i = x - xmin + 1
            mps.data[i] *= peps.tensors[x, y]
        end
        # normalize
        for i = 1:length(mps.data)
            h = Int(floor(log2(LinearAlgebra.norm(mps.data[i]))))
            mps.data[i] /= exp2(h)
            expval += h
        end
        # truncate
        truncate!(mps, maxdim=sc_bond_dim, cutoff=svd_eps)
        # normalize
        for i = 1:length(mps.data)
            h = Int(floor(log2(LinearAlgebra.norm(mps.data[i]))))
            mps.data[i] /= exp2(h)
            expval += h
        end
    end

    # get output
    T = ITensor([1.])
    for T2 in mps.data
        T *= T2
    end
    val = storage(T)[1]
    h = Int(floor(log2(LinearAlgebra.norm(val))))
    val /= exp2(h)
    expval += h

    return val, expval
end

function diag_inv(A::Matrix)
    @assert size(A, 1) == size(A, 2)
    d = size(A, 1)
    B = copy(A)
    for i = 1:d
        B[i, i] = inv(A[i, i])
    end
    return B
end

function renormalize_tensor!(net::ContractionNetwork{P}, site::P) where P
    h = Int(floor(log2(LinearAlgebra.norm(net[site]))))
    net[site] /= exp2(h)
    return h
end

function renormalize_bond_matrix!(net::ContractionNetwork{P}, site1::P, site2::P) where P
    h = Int(floor(log2(LinearAlgebra.norm(net.bond_matrices[Set([site1, site2])]))))
    net.bond_matrices[Set([site1, site2])] /= exp2(h)
    return h
end

function canonicalness(net::ContractionNetwork{P}) where P
    f = 0.0

    for (site1, v1) in net.tensors
        for site2 in neighbors(net, site1)

            # compute isometry
            iso = net[site1]
            for site3 in neighbors(net, site1)
                if site3 != site2
                    iso *= bond_matrix(net, site1, site3)
                end
            end
            # compute contraction of isometry with itself
            idx = commonind(iso, bond_matrix(net, site1, site2))
            idx2 = Index(dim(idx))
            iso_prime = iso * delta(idx, idx2)
            T = iso * iso_prime

            # compare with identity
            LHS = T / sum(diag(T))
            RHS = dense(delta(idx, idx2))
            RHS /= sum(diag(RHS))
            f += 0.5 * norm(LHS - RHS)
        end

    end

    return f / length(net.bond_matrices)
end