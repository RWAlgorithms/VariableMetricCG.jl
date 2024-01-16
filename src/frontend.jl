function batchfit(
    constraints::ConstraintsContainer,
    metrics::MetricsContainer,
    f,
    fdf!,
    p0s::Vector{Vector{T}},
    RCG_config::RCGConfig,
    ) where T

    M = length(p0s)
    results = Vector{Solution{T}}(undef, M)

    for m in eachindex(p0s)
        results[m] = runRCG!(
            constraints,
            metrics,
            f,
            fdf!,
            p0s[m],
            RCG_config,
        )
    end

    return results
end

@kwdef struct RCGCompactConfig{T}

    ρ_C::T = convert(T, 0.5)
    ρ1::T = convert(T, 0.8)
    ρ2::T = convert(T, 0.8)
    step_shrink_rate::T = convert(T, 0.9)
    max_iters::Int = 1000
    line_search_max_iters::Int = 1000
    μ_min::T = convert(T, 0.3)
    μ_max::T = convert(T, 20)
end

function getOviedo2013config(
    ::Type{T};
    max_iters::Int = 100_000,
    line_search_max_iters::Int = 1000,
    ) where T

    return RCGCompactConfig{T}(
        max_iters = max_iters,
        line_search_max_iters = line_search_max_iters,
        ρ_C = convert(T, 0.85),
        ρ1 = convert(T, 1e-4),
        ρ2 = convert(T, 1e-4),
        step_shrink_rate = convert(T, 0.2),
        μ_min = convert(T, 1e-10),
        μ_max = convert(T, 10e10),
    )
end

# for those who just want a minimum hassle bounded CG.
function minimizeoverbox(
    f,
    fdf!,
    lbs::Vector{T},
    ubs::Vector{T},
    C::RCGCompactConfig{T};
    p_initial::Vector{T} = (lbs+ubs) ./ 2,
    grad_tol::T = convert(T, 1e-6),
    ) where T <: AbstractFloat

    N_vars = length(lbs)
    @assert N_vars == length(ubs)

    constraints = ConstraintsContainer(
        BoxConstraints(lbs, ubs, 1:N_vars),
    )

    initial_min_step = convert(T, grad_tol * 1e-3) # for BB-step.

    RCG_config = RCGConfig(
        grad_tol,
        BonnettiniBB(
            BarzilaiBorweinBuffer(
                initial_min_step,
                N_vars,
            );
            ζ = convert(T, 2.0),
            τ_initial = one(T),
            window_length = 30,
        ),
        NoRestart(), # objective is highly multi-modal; switch to Bayesian opt.
        FollowStepμ(C.μ_min, C.μ_max),
        nothing;
        ρ_C = C.ρ_C, 
        ρ1 = C.ρ1, 
        ρ2 = C.ρ2, 
        step_shrink_rate = C.step_shrink_rate, 
        max_iters = C.max_iters,
        line_search_max_iters = C.line_search_max_iters,
        t_lb = max(grad_tol, eps(T)*2),
    )

    result = runRCG!(
        constraints,
        getMetricsContainer(
            nothing,
            ConstantMetric(one(T), 1:N_vars),
        ),
        f,
        fdf!,
        p_initial,
        RCG_config,
    )

    return result
end

function extractminimizer(R::Solution{T}) where T <: AbstractFloat
    return R.p_min
end

function extractminimum(R::Solution{T}) where T <: AbstractFloat
    return R.f_p_min
end