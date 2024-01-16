
### read, update, reset the line search parameters.


# initialize with C_0, Q_0.
function resetlinesearchparams!(A::Oviedo2022LS{T}, f_p0::T) where T
    
    A.mutables.C = f_p0
    A.mutables.Q = one(T)

    return nothing
end

function readC(A::Oviedo2022LS{T})::T where T
    return A.mutables.C
end

function readQ(A::Oviedo2022LS{T})::T where T
    return A.mutables.Q
end

function writeC!(A::Oviedo2022LS{T}, x::T) where T
    A.mutables.C = x
    return nothing
end

function writeQ!(A::Oviedo2022LS{T}, x::T) where T
    A.mutables.Q = x
    return nothing
end

# assumes the inputs are finite numbers.
function updatelinesearchparams!(A::Oviedo2022LS{T}, f_p::T) where T

    ρ_C = A.ρ_C

    C_prev = readC(A)
    Q_prev = readQ(A)
    
    # update C, Q for the next iteration.
    Q = ρ_C*Q_prev + one(T)
    C = (ρ_C*Q_prev*C_prev + f_p)/Q
    
    writeC!(A, C)
    writeQ!(A, Q)

    return nothing
end

##### line search from Oviedo 2022.


function computeCGdir!(
    vars::Variables{T},
    vector_norms::VectorNorms{T},
    config::Oviedo202CG,
    ) where T


    proj_z_norm_prev = vector_norms.proj_z

    # get the scaled vector transport VT(zZ,Z) = VT(tU, U*norm(Z)_prev_iterate), Z := proj_z.
    factor1 = proj_z_norm_prev/vector_norms.u_t
    VT_aZ_Z = vars.u_t .* factor1

    μ = getμ(config.μ_strategy)
    β = -μ*dot(vars.df, VT_aZ_Z)/proj_z_norm_prev^2

    vars.z[:] = -vars.df♯ + β .* VT_aZ_Z # partial update, without the df♯ at p_t.

    return nothing
end

function getμ(A::Constantμ{T})::T where T
    return A.μ
end

function getμ(A::FollowStepμ{T})::T where T
    return A.step_size[begin]
end

##################

function updateCGhyperparams!(
    config::Oviedo202CG{FollowStepμ{T},CGV},
    R::LineSearchSolution{T},
    vector_norms::VectorNorms{T},
    ) where {T, CGV <: CGVariants}

    config.μ_strategy.step_size[begin] = R.t * vector_norms.negative_df♯
    
    return nothing
end

function updateCGhyperparams!(
    ::Oviedo202CG{Constantμ{T},CGV},
    args...
    ) where {T, CGV <: CGVariants}

    return nothing
end

##################


# The possible return status: 
# :reached_max_line_search_iters 
# :min_step_and_boundary # this one means possible intersection with boundary. check tagent cone conditions.
# :min_step
# :success
#
function linesearch!(
    config::Oviedo2022LS{T}, # mutating.
    diagnostics, # mutates if storing values.
    constraints::ConstraintsContainer,
    t_initial::T,
    f,
    vars::Variables,
    v_norms::VectorNorms;
    verbose::Bool = false,
    )::LineSearchSolution{T} where T

    vars.u_t[:] = vars.u # Use straight-line retraction: parallel transport is identity. Remove this if using non-straight line retraction. In that case, need to solve for this every time we evaltrajectory in linesearch!()

    # unpack.
    uf = v_norms.dir_derivative_u
    u = vars.u
    p = vars.p
    p_t = vars.p_t # buffer to hold result of candidate position.
    
    t_lb = config.t_lb
    ρ_C, ρ1, ρ2, step_shrink_rate = config.ρ_C, config.ρ1, config.ρ2, config.step_shrink_rate
    C = readC(config)
    #@show C, ρ_C, ρ1, ρ2, uf

    term1 = ρ1*uf
    #term2 = -ρ2*z_norm_sq
    term2 = -ρ2

    # # for diagnostics when we don't find solution in max_iters.
    min_LHS = convert(T, Inf)
    min_t = zero(T) # min_t is the t that had the smallest LHS so far.
    LHS = convert(T, NaN)

    # # initialize. reduce t_initial until box bounds are satisfied.
    
    t = t_initial

    # evaltrajectory!
    updateiterate!(p_t, t, u, p)
    
    valid_p_t = satisfyconstraints(constraints, p_t)

    while !valid_p_t && t >= config.t_lb
        t = t*step_shrink_rate

        # evaltrajectory!
        updateiterate!(p_t, t, u, p)
        valid_p_t = satisfyconstraints(constraints, p_t)
    end


    if !valid_p_t
        # minimum step rate does not yield valid step.
        return LineSearchSolution(
            t,
            LHS,
            p_t, # evaltrajectory!() updates blocks, which is part of vars.
            t_initial,
            false,
            :cannot_find_valid_initial_t,
        )
    end
    
    # # non-monotone line search.
    for _ = 1:config.max_iters

        RHS = C + term1*t + term2*t^2 # eq:line_search_condition_unit_vector in RCG notes.

        # update and check p_t against constraints and stopping condition.
        # evaltrajectory!
        updateiterate!(p_t, t, u, p)
        valid_p_t = satisfyconstraints(constraints, p_t)

        if valid_p_t
            LHS = f(p_t)
            updatediagnostics!(diagnostics, LHS, p_t)

            #@show LHS, RHS, abs(LHS-RHS)
            if LHS <= RHS
                #@show t_initial, debug_counter, t, LHS, RHS # debug.
                return LineSearchSolution(
                    t,
                    LHS,
                    p_t, # evaltrajectory!() updates blocks, which is part of vars.
                    t_initial,
                    true,
                    :success,
                )
            end
        end

        # shrink step size.
        t = t*step_shrink_rate
        if abs(t) < t_lb
            return LineSearchSolution(
                t,
                LHS,
                p_t, # evaltrajectory!() updates blocks, which is part of vars.
                t_initial,
                false,
                :t_lb_reached,
            )
        end

        if min_LHS > LHS
            min_LHS = LHS
            min_t = t
        end
    end

    # cannot find a step that satisfies the line search condition.
    # return the step with lowest score instead.
    if verbose
        println("max line search iters reached. Used the t that had the smallest LHS for p_t in LineSearchSolution.")
        println("The last t's LHS was: ", LHS)
    end

    # for diagnostic purposes: update p_t with min_t before exiting. min_t is the t that had the smallest LHS so far.
    # evaltrajectory!
    updateiterate!(p_t, min_t, u, p)

    return LineSearchSolution(
        min_t,
        min_LHS,
        p_t, # evaltrajectory!() updates blocks, which is part of vars.
        t_initial,
        false,
        :max_line_search_iters,
    )
end

function updateiterate!(p_t::Vector{T}, t::T, u::Vector{T}, p::Vector{T})::Nothing where T
    @assert length(p_t) == length(p) == length(u)

    p_t[:] = p
    axpy!(t, u, p_t)

    return nothing
end