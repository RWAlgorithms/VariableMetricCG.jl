
## line search config.

abstract type LineSearchConfig end

mutable struct Oviedo2022buffer{T}
    C::T
    Q::T
end

# line search from (Oviedo 2022).
struct Oviedo2022LS{T} <: LineSearchConfig
    ρ_C::T
    ρ1::T
    ρ2::T
    step_shrink_rate::T
    max_iters::Int
    
    # this is the normalized magnitude gradient.
    #min_step_norm::T # now t_lb
    #max_step_norm::T # now t_ub
    #max_iterate_difference::T # now t_ub
    
    t_lb::T # wrt the unit vector version of vars.proj_z, vars.u.
    t_ub::T # wrt the unit vector version of vars.proj_z, vars.u.

    # parameters or buffers that are updted throughout the optimization.
    mutables::Oviedo2022buffer{T}
end



#######################3
abstract type CGVariants end

# if encountered binding constraints at an iterate, use only steepest descent as the search direction. This removes history of the CG search direction.
struct Restart <: CGVariants end
struct NoRestart <: CGVariants end # use the CG formula even at binding constraints.

##############################
abstract type CGμStrategy end

struct Constantμ{T} <: CGμStrategy
    μ::T
end

struct FollowStepμ{T} <: CGμStrategy
    μ_min::T
    μ_max::T

    # single-element for mutating.
    step_size::Vector{T}
end

function FollowStepμ(lb::T, ub::T)::FollowStepμ{T} where T
    return FollowStepμ(lb, ub, ones(T,1))
end


###################
abstract type CGConfig end

struct Oviedo202CG{MT <: CGμStrategy, CGV<:CGVariants} <: CGConfig
    #μ::T
    μ_strategy::MT
    variant::CGV
end

################## Oviedo's RCG.

#### Initial step size.
abstract type InitialStepStrategy end
# see RBB.jl for subtypes.

#### optimization config. This is a combination of the previous configs.

abstract type DiagnosticsType end

struct StoreEvals{T <: AbstractFloat} <: DiagnosticsType
    p_set::Vector{Vector{T}}
    f_p_set::Vector{T}
end

function StoreEvals(::Type{T}) where T <: AbstractFloat
    return StoreEvals(Vector{Vector{T}}(undef, 0), Vector{T}(undef, 0))
end

function resetbuffer!(A::StoreEvals{T}) where T
    resize!(A.p_set, 0)
    resize!(A.f_p_set, 0)
    return nothing
end

function resetbuffer!(::Nothing)
    return nothing
end

#struct NoDiagnostics <: DiagnosticsType end # just use Nothing.

function updatediagnostics!(::Nothing, args...)
    return nothing
end

function updatediagnostics!(A::StoreEvals{T}, f_p::T, p::Vector{T}) where T
    push!(A.p_set, copy(p))
    push!(A.f_p_set, f_p)
    return nothing
end

# # the master config.
struct RCGConfig{
    T <: AbstractFloat,
    LT <: LineSearchConfig,
    CGT <: CGConfig,
    BT <: InitialStepStrategy,
    DT <: Union{DiagnosticsType, Nothing}
    }
    
    grad_tol::T
    max_iters::Int

    #tangent_cone_config::TangentConeConfig{T}
    cg_config::CGT
    #trajectory_config::PT # PowerSeriesIVP-related config.
    line_search_config::LT

    initial_step_strategy::BT

    diagnostics::DT

    verbose::Bool
end


function RCGConfig(
    grad_tol::T,
    initial_step_strategy::ST,
    CG_variant::CGV,
    μ_strategy::MCG,
    diagnostics::DT;
    verbose::Bool = true,
    ρ_C::T = convert(T, 0.5), # η in paper.
    ρ1::T = convert(T, 0.8),
    ρ2::T = convert(T, 0.8),
    step_shrink_rate::T = convert(T, 0.9),
    max_iters::Int = 1000,
    line_search_max_iters::Int = 1000,
    
    # this is the normalized magnitude gradient.
    #t_lb::T = max(grad_tol, eps(T)*100 ), # wrt the unit vector version of vars.proj_z, vars.u.
    t_lb::T = max(grad_tol, eps(T)*100), # wrt the unit vector version of vars.proj_z, vars.u.
    t_ub::T = convert(T, 100), # wrt the unit vector version of vars.proj_z, vars.u.
    ) where {T, CGV<:CGVariants, MCG, ST <: InitialStepStrategy, DT <: Union{DiagnosticsType, Nothing}}

    cg_config = Oviedo202CG(μ_strategy, CG_variant)
    line_search_config = Oviedo2022LS(
        ρ_C,
        ρ1,
        ρ2,
        step_shrink_rate,
        line_search_max_iters,
        t_lb,
        t_ub,
        Oviedo2022buffer(convert(T, NaN), one(T)), # need to run resetlinesearchparams!() first time we have a valid objective function eval.
    )
    
    return RCGConfig(
        grad_tol,
        max_iters,

        #tangent_cone_config,
        cg_config,
        #trajectory_config,
        line_search_config,
        initial_step_strategy,
        diagnostics,
        verbose,
    )
end

# ### quantities in here used for diagnostics.
struct LineSearchSolution{T}

    t::T
    f_p_t::T

    p_t::Vector{T}

    # diagnostics.
    t_initial::T
    
    successful::Bool
    status::Symbol
end

