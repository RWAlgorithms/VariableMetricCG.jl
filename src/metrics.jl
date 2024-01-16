


function updatemetrictensor!(G_vec::Vector{T}, C::MetricsContainer, p::Vector{T}, ) where T

    for m in eachindex(C.rational22)
        updatemetrictensor!(G_vec, C.rational22[m], p)
    end

    for m in eachindex(C.constant)
        updatemetrictensor!(G_vec, C.constant[m], p)
    end

    return nothing
end

function updatemetrictensor!(::Vector, ::Nothing, args...)
    return nothing
end

function updatemetrictensor!(
    G_vec::Vector{T},
    metric::Rational22{T,MT},
    p::AbstractVector,
    ) where {T, MT <: Union{Int, UnitRange{Int}, Vector{Int}}}

    a_sq = metric.a^2
    b_sq = metric.b^2
    multiplier = metric.multiplier

    # see diagonal notes, section on pair-wise notch.

    x = p[metric.mapping] # TODO make this non-allocating.
    
    for i in eachindex(x)
        diff = sum( (x[i]-x[k])^2 for k in eachindex(x) )
        g_ii = (a_sq + diff)/(b_sq + diff)

        k = metric.mapping[i]
        G_vec[k] = g_ii*multiplier
    end

    return nothing
end


function updatemetrictensor!(
    G_vec::Vector{T},
    metric::ConstantMetric{T,MT},
    args...
    ) where {T, MT <: Union{Int, UnitRange{Int}, Vector{Int}}}

    for k in metric.mapping
        G_vec[k] = metric.a
    end

    return nothing
end

# function updatemetrictensor!(::Vector{T}, C::EuclideanMetric, args...) where T <: AbstractFloat
#     for k in metric.mapping
#         G_vec[k] = one(T)
#     end
#     return nothing
# end

# function updatemetrictensor!(::Vector, ::EuclideanMetric, args...)
#     return nothing
# end

############# update gradient sharp.


function computedfsharp!(df♯::Vector{T}, df::Vector{T}, G_vec::Vector{T}) where T
    
    @assert length(df♯) == length(G_vec) == length(df)

    for i in eachindex(G_vec)   
        df♯[i] = df[i] / G_vec[i] # inverse metric tensor to get sharp isomorphism.
    end

    return nothing
end

# dot(v, G_vec .* v)
function evalnormsq(G_vec::Vector{T}, v::Vector{T}) where T
    return sum( G_vec[i]*v[i]^2 for i in eachindex(v) )
end