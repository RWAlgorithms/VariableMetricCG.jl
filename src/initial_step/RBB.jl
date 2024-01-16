# use (Iannazzo 2018), (Crisci 2020), (Bonettini 2009)

######### step size

################

# (Iannazzo 2018).
struct BarzilaiBorweinBuffer{T}
    min_step::T

    # intermediate outputs.
    # Let g be the current gradient♯. gp be the previous gradient♯
    # Let Tgp be the vector transported previous gradient♯.
    s_prev::Vector{T} # -step_prev*Tgp. Eq. 4.5.
    y_prev::Vector{T} # g - Tgp. Eq.4.6.
end

function BarzilaiBorweinBuffer(min_step::T, D::Integer)::BarzilaiBorweinBuffer{T} where T
    return BarzilaiBorweinBuffer(
        min_step,
        Vector{T}(undef, D),
        Vector{T}(undef, D),
    )
end

#struct UseBB1 <: InitialStepStrategy end
#struct UseBB2 <: InitialStepStrategy end
struct BonnettiniBB{T} <: InitialStepStrategy
    BB2s::Vector{T}

    ζ::T # larger than 1.
    τ_initial::T # larger than 0.

    # single-element array to avoid mutable struct.
    τ::Vector{T}
    update_index::Vector{Int} # for BB2s.

    buffer::BarzilaiBorweinBuffer{T}
end

function BonnettiniBB(
    buffer::BarzilaiBorweinBuffer{T};
    ζ::T = convert(T, 2.0),
    τ_initial::T = one(T),
    window_length::Int = 10,
    )::BonnettiniBB{T} where T

    @assert ζ > one(T)
    @assert zero(T) < τ_initial #< one(T)

    return BonnettiniBB(
        collect( convert(T, Inf) for _ = 1:window_length ),
        ζ,
        τ_initial,
        collect( convert(T, 0.9) for _ = 1:1 ),
        zeros(Int, 1),
        buffer,
    )
end

# circularbuffer index update.
# the logic comes from: collect(mod(i-1,5)+1 for i = 1:10) for a length 5 BB2s.
function updateindex!(B::BonnettiniBB)

    ind = B.update_index[begin]
    B.update_index[begin] = mod(ind+1-1, length(B.BB2s))+1

    return nothing
end

function updateBB2!(B::BonnettiniBB{T}, BB2::T) where T
    updateindex!(B)

    B.BB2s[B.update_index[begin]] = BB2

    return nothing
end

##########

function updateBBdependecies!(
    A::BarzilaiBorweinBuffer{T},
    df♯::Vector{T},
    VT_df♯_prev::Vector{T},
    u_t::Vector{T},
    ) where T

    #A.s_prev[:] = -prev_step .* VT_df♯_prev
    A.s_prev[:] = u_t
    A.y_prev[:] = df♯ .- VT_df♯_prev

    return nothing
end

# Eq. 1.6. of (Crisci 2020), which came from (Bonettini 2009)
function computeinitialstep!(
    #BB_buffer::BarzilaiBorweinBuffer{T}, # mutates.
    strategy::BonnettiniBB, # mutates.
    df♯::Vector{T},
    VT_df♯_prev::Vector{T},
    u_t::Vector{T},
    )::T where T
    
    BB_buffer = strategy.buffer
    τ = strategy.τ[begin]
    #BB2s = strategy.BB2s
    ζ = strategy.ζ

    # A.steps[begin+1] = minimum(BB.BB2s)
    updateBBdependecies!(BB_buffer, df♯, VT_df♯_prev, u_t)
    s_prev = BB_buffer.s_prev
    y_prev = BB_buffer.y_prev

    # take abolute values.
    BB1 = abs( dot(s_prev, s_prev)/dot(s_prev, y_prev) )
    BB2 = abs( dot(s_prev, y_prev)/dot(y_prev, y_prev) )
    
    # Bonnettini strategy.
    
    # the BB1 case.
    initial_step_size = BB1
    strategy.τ[begin] = τ*ζ # update strategy parameters.

    # check if we should use the BB2 case.
    if BB2/BB1 < τ
        
        updateBB2!(strategy, BB2)
        initial_step_size = minimum(strategy.BB2s)

        strategy.τ[begin] = τ/ζ # update strategy parameters.
    end

    return initial_step_size
end
# quick intro to BB step: https://www.cmor-faculty.rice.edu/~yzhang/caam565/L1_reg/BB-step.pdf

# implements step ∈ [step_min, step_max], but wrt a unit norm step direction, vars.u instead of the search dir vector vars.proj_z
function getinitialtime(
    vars::Variables{T},
    v_norms::VectorNorms{T},
    initial_step_size::T,
    config::Oviedo2022LS{T};
    verbose::Bool = false,
    )::T where T

    # check if proj_z is finite. if not, reset search dir to negative gradient.
    if !all( isfinite(vars.proj_z[d]) for d in eachindex(vars.proj_z) )
        if verbose
            println("Warning: non-finite proj_z detected. Reset proj_z to -df♯.")
        end
        vars.proj_z[:] = -vars.df♯
        v_norms.proj_z = v_norms.negative_df♯
    end

    # # get unit vector and its step, aka t_fin.
    # we want t_fin .* u == step.
    vars.u[:] = vars.proj_z ./ v_norms.proj_z

    a = initial_step_size* v_norms.proj_z

    if !isfinite(a) || a < 0
        # assign it any valid max step size. We use the mid point of the config bound values.
        a = (config.t_lb + config.t_ub)/2
    end

    t_fin = clamp(a, config.t_lb, config.t_ub)

    return t_fin
end

# function getinitialtime(
#     t0::T,
#     vector_norms::VectorNorms,
#     config::Oviedo2022LS{T},
#     )::T where T

#     if !isfinite(t0) || t0 < config.t_lb || t0 > config.t_ub
        
#         t = one(T)/vector_norms.negative_df♯ # make the initial step a unit vector.
#         t = clamp(t, config.t_lb, config.t_ub)
#         return t
#     end

#     return t0
# end

