
##### projection to tangent cone.

function project2tangentcone!(
    proj_z::Vector{T},
    z::Vector{T},
    p::Vector{T},
    C::ConstraintsContainer,
    ) where T <: Real

    projectTCbox!(proj_z, C.box, z, p)
    return nothing
end

function projectTCbox!(::Vector, ::Nothing, args...)
    return nothing
end

# project a vector onto the tangent cone.
function projectTCbox!(
    proj_x::Vector{T}, # cvan be x. mutates.
    B::BoxConstraints{T,MT},
    x::Vector{T},
    p::Vector{T},
    ) where {T, MT <: Union{Nothing, Int, UnitRange{Int}, Vector{Int}}}
    
    @assert length(proj_x) == length(x) == length(p)

    for i in eachindex(x)

        if B.lbs[i] < p[i] < B.ubs[i]
            # interior case.
            proj_x[i] = x[i]
        else
            # outside or boundary case.

            # we could have lb == ub, so we need separate if-statements.
            if !(B.lbs[i] < p[i])
                proj_x[i] = activelb(x[i])
            end
            
            if !(p[i] < B.ubs[i])
                proj_x[i] = activeub(x[i])
            end
        end
    end

    return nothing
end

function activelb(x::T)::T where T
    if x < 0 # zero decreasing rate of change, since we're at the lower bound already.
        return zero(T)
    end

    return x
end

function activeub(x::T)::T where T
    if x > 0 # zero increasing rate of change, since we're at the upper bound already.
        return zero(T)
    end

    return x
end

# check if t .* p satisfy the box constraints.
function satisfyconstrainttype(
    C::BoxConstraints{T,MT},
    p_t::Vector{T},
    )::Bool where {T, MT <: Union{Int, UnitRange{Int}, Vector{Int}}}

    for i in C.mapping
        #if p_t[i] < C.lbs[i] || p_t[i] > C.ubs[i]
        if !(C.lbs[i] <= p_t[i] <= C.ubs[i]) # boundary allowed.
            return false
        end
    end

    return true
end

function satisfyconstraints(C::ConstraintsContainer, args...)::Bool
    status = satisfyconstrainttype(C.box, args...)
    return status
end

function satisfyconstrainttype(::Nothing, args...)::Bool
    return true
end