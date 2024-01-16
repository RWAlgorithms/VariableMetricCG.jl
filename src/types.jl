
abstract type MetricType end

struct Rational22{T <: AbstractFloat, MT <: Union{Int, UnitRange{Int}, Vector{Int}}} <: MetricType
    a::T
    b::T
    multiplier::T
    mapping::MT
end

function createmetric(template::Rational22{T}, mapping::MT)::Rational22{T, MT} where {T,MT}
    return Rational22(template.a, template.b, template.multiplier, mapping)
end

struct ConstantMetric{T <: AbstractFloat, MT <: Union{Int, UnitRange{Int}, Vector{Int}}} <: MetricType
    a::T
    mapping::MT
end

function createmetric(template::ConstantMetric{T}, mapping::MT)::ConstantMetric{T, MT} where {T,MT}
    return ConstantMetric(template.a, mapping)
end

# struct EuclideanMetric{MT <: Union{Int, UnitRange{Int}, Vector{Int}}}  <: MetricType
#     mapping::RT
# end

# don't export this. Update with new types of metricds in the future.
struct MetricsContainer{
    RST <: Union{Rational22, Nothing},
    CT <: Union{ConstantMetric, Nothing},
    }

    # ::Nothing means no variable use the type of metric.
    rational22::Vector{RST}
    constant::Vector{CT}
    #EuclideanMetric::ET # default to Euclidean metric.
end

## export these:
function getMetricsContainer(r::RST)::MetricsContainer{RST, Nothing} where RST <: Rational22
    return MetricsContainer([r;], Vector{Nothing}(undef, 0))
end

function getMetricsContainer(c::CT)::MetricsContainer{CT, Nothing} where CT <: ConstantMetric
    return MetricsContainer(Vector{Nothing}(undef, 0), [c;])
end

# function getMetricsContainer(u::ET)::MetricsContainer{ET, Nothing} where ET <: EuclideanMetric
#     return MetricsContainer(nothing, nothing, u)
# end

function getMetricsContainer(r::RST, c::CT)::MetricsContainer{RST,CT} where {RST, CT}
    return MetricsContainer([r;], [c;])
end

########### 

abstract type Constraints end

struct BoxConstraints{T, MT <: Union{Nothing, Int, UnitRange{Int}, Vector{Int}}} <: Constraints
    lbs::Vector{T}
    ubs::Vector{T}
    mapping::MT # ::Nothing means there are no box constrained variables.
end

# struct UnConstrained{MT <: Union{Nothing, Int, UnitRange{Int}, Vector{Int}}} <: Constraints
#     mapping::RT # ::Nothing means all variables are unconstrained.
# end

# export constructor with keyboards. Update with new types of constraints in the future.
struct ConstraintsContainer{BT <: Union{BoxConstraints, Nothing}}

    box::BT # BoxConstraints can store multiple box constraints.
    # inequalities::Vector{IT} # multiple inequality constraints.

    # the variables that were left alone by box would default to being unconstrained.
    #unconstrained::UT
end


############ RCG engine containers.


struct Variables{T}
    p::Vector{T}
    p_t::Vector{T}
    
    df♯_t::Vector{T} # this is T(k->k+1)(g_k) in Eqs. 4.5, 4.6 of Iannazzo 2018.
    df♯::Vector{T} # This is g_{k+1}. Transport this for BB step of the next iter.
    df::Vector{T}

    # RCG's z update.
    z::Vector{T}
    proj_z::Vector{T}
    
    u::Vector{T} # unit vector version of proj_z.
    u_t::Vector{T} # This is T(t*a_k)(z_k) in Eq. 29 of Oviedo 2022, without the non-expansive condition.

end

function Variables(p::Vector{T}) where T <: AbstractFloat
    return Variables(p, similar(p), similar(p), similar(p), similar(p), similar(p), similar(p), similar(p), similar(p))
end

mutable struct VectorNorms{T}
    
    # all updated in RCG() loop.
    negative_df♯::T # after df!()

    # after computeCGdir!()
    proj_z::T # actually stores the previous iterate's z when this is used in computeCGdir!().
    #proj_z_prev::T # norm(proj_z_p_{k-1})_{p_{k-1}}.
    
    # before updatelinesearchparams!()
    dir_derivative_u::T # directional derivative in dir of proj_z evaluated at p.

    # after linesearch!()
    u_t::T
end

function VectorNorms(::Type{T})::VectorNorms{T} where T
    return VectorNorms(one(T), one(T), one(T), one(T))
end

struct Solution{T}
    p_initial::Vector{T} # for diagnostics.
    f_p_initial::T

    p::Vector{T}
    df::Vector{T}
    #df♯::Vector{T}
    f_p::T
    
    p_min::Vector{T}
    f_p_min::T

    iters_ran::Int
    vector_norms::VectorNorms{T}
    
    status::Symbol
    #line_search_sol::LineSearchSolution{T}

    # diagnostics::Diagnostics{T}
end