
# 
"""
runRCG!(
    constraints::ConstraintsContainer,
    metrics::MetricsContainer,
    f,
    fdf!,
    p_initial::Vector{T}, # not mutated.
    config::RCGConfig,
    ) where T
    
Assumes f cannot evaluate to NaN.

possible exit conditions
:success
so on.
"""
function runRCG!(
    constraints::ConstraintsContainer,
    metrics::MetricsContainer,
    f,
    fdf!,
    p_initial::Vector{T}, # not mutated.
    config::RCGConfig,
    ) where T
    
    max_iters = config.max_iters

    #trajectory_buffer, records, Gs = setupRCGbuffers(vars)
    #projection_status = records.projection
    #retraction_status = records.retraction

    #blocks = vars.blocks
    #v_norms = vars.vector_norms
    
    N_vars = length(p_initial)

    # essential initialization for the initial iteration.
    G_vec = ones(T, N_vars) # diagonal metric. Default to Euclidean metric.

    vars = Variables(copy(p_initial))
    fill!(vars.u_t, zero(T))

    v_norms = VectorNorms(one(T), one(T), one(T), zero(T))
    v_norms.proj_z = one(T)
    v_norms.u_t = one(T)

    # allocate.
    f_p::T = f(vars.p)
    resetlinesearchparams!(config.line_search_config, f_p)

    f_p_initial = f_p
    
    # misc initialization.
    fill!(vars.df, one(T))
    fill!(vars.df♯_t, convert(T,1+1/N_vars) )
    
    p_min = copy(p_initial)
    f_p_min::T = f_p

    # optim.
    for n = 1:max_iters
        
        # # Step: Check binding constraints. # skip since we're doing box constraints.
        # checkconstraints!(
        #     constraints,
        #     UpdateIndicators(),
        #     #UseIterate(),
        #     blocks,
        #     config.tangent_cone_config,
        # )
        
        # # Step: get gradient and objective.
        # Updates df in vars.
        f_p = fdf!(vars.df, vars.p)
        # if vars.df[end-1] < 0
        #     @show vars.df[end-1]
        # end
        #project2tangentcone!(vars.df, vars.df, vars.p, constraints)

        if f_p < f_p_min
            p_min[:] = vars.p
            f_p_min = f_p
        end

        if !isfinite(f_p) || !(all(isfinite.(vars.df)))
            return packagesol(
                vars,
                p_initial,
                p_min,
                f_p_min,
                f_p,
                f_p_initial,
                n,
                v_norms,
                :non_finite_objective_or_gradient,
            )
        end

        # # Step: Sharp-isomorphism.
        # Updates df♯, norm(-df♯).
        updatemetrictensor!(G_vec, metrics, vars.p)
        computedfsharp!(vars.df♯, vars.df, G_vec)
        
        #projectnegativedfsharp!(blocks, projection_status, constraints)

        v_norms.negative_df♯ = sqrt(evalnormsq(G_vec, -vars.df♯))
        
        # # Step: get search direction.
        computeCGdir!(vars, v_norms, config.cg_config) # Updates z
        project2tangentcone!(vars.proj_z, vars.z, vars.p, constraints) # Updates proj_z
        v_norms.proj_z = sqrt(evalnormsq(G_vec, vars.proj_z))
        
        # check stopping conditions.
        if v_norms.negative_df♯ < config.grad_tol || v_norms.proj_z < config.grad_tol
            return packagesol(
                vars,
                p_initial,
                p_min,
                f_p_min,
                f_p,
                f_p_initial,
                n,
                v_norms,
                :success,
            )
        end
      

        # # get initial step size.
        # initial_step_size is wrt proj_z, not u := proj_z/v_norms.proj_z.
        initial_step_size = computeinitialstep!(
            config.initial_step_strategy,
            vars.df♯,
            vars.df♯_t,
            vars.u_t,
        )
        # t is wrt u. Updates u.
        #t_BB = getinitialtime!(vars, initial_step_size, config.line_search_config) # updates u.
        # simplify this by just bounding it like in Oviedo 2022.
        #t_initial = getinitialtime(initial_step_size, v_norms, config.line_search_config)
        t_initial = getinitialtime(
            vars,
            v_norms,
            initial_step_size,
            config.line_search_config,
        )
        project2tangentcone!(vars.u, vars.u, vars.p, constraints)

        # # # Step: retraction. # no longer used since we use straight lines.
        # # updates the vector buffers p_0, u_0, df♯_0 for each block.
        # sols = gettrajectory!(
        #     retraction_status,
        #     blocks,
        #     projection_status,
        #     metrics,
        #     constraints_ivp,
        #     t_BB, 
        #     #config.trajectory_config,
        #     ivp_config,
        # ) # handles projection and call PowerSeriesIVP to handle intersection.

        # # Step: line search.
        # Updates p_t, u_t, df♯_t.

        #v_norms.dir_derivative_u = evaldiruderivative(vars)
        v_norms.dir_derivative_u = dot(vars.df, vars.u)

        updatelinesearchparams!(config.line_search_config, f_p)

        ls_results = linesearch!(
            config.line_search_config,
            config.diagnostics,
            constraints,
            t_initial,
            f,
            vars,
            v_norms,
        )
        v_norms.u_t = sqrt(evalnormsq(G_vec, vars.u_t))

        # update CG config so μ can utilize the line search results.
        updateCGhyperparams!(config.cg_config, ls_results, v_norms)
        
        if !ls_results.successful

            # exit if we failed line search.
            return packagesol(
                vars,
                p_initial,
                p_min,
                f_p_min,
                f_p,
                f_p_initial,
                n,
                v_norms,
                ls_results.status,
            )
        
        elseif !(all(isfinite.(vars.p_t)))

            return packagesol(
                vars,
                p_initial,
                p_min,
                f_p_min,
                f_p,
                f_p_initial,
                n,
                v_norms,
                :accepted_non_finite_iterate,
            )
        end

        # update iterate. 
        # we assume linesearch!() gets us p_t that satisfy constraints, numerically.
        vars.p[:] = vars.p_t[:]
    end

    return packagesol(
        vars,
        p_initial,
        p_min,
        f_p_min,
        f_p,
        f_p_initial,
        max_iters,
        v_norms,
        :max_iter_reached,
    )
end

function packagesol(
    vars::Variables{T},
    p_initial::Vector{T},
    p_min::Vector{T},
    f_p_min::T,
    f_p::T,
    f_p_initial::T,
    iter::Int,
    v_norms::VectorNorms{T},
    exit_status::Symbol,
    #line_search_sol::LineSearchSolution{T},
    )::Solution{T} where T
    
    return Solution(
        p_initial,
        f_p_initial,

        vars.p,
        vars.df,
        f_p,

        p_min,
        f_p_min,
        
        iter,
        v_norms,
        exit_status,
        #line_search_sol,
    )
end

