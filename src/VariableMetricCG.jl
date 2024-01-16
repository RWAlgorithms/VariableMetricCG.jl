module VariableMetricCG

using LinearAlgebra

include("types.jl")
include("./RCG/CG_types.jl") # TODO split this up further into the right folders. Create new folders if we need to. Separate the types from the methods.

# projection onto constraints or tangent cone.
include("project.jl")

# metrics.
include("metrics.jl")

# line search.
include("./RCG/line_search.jl")

# initial step size.
include("initial_step/RBB.jl")

# algorithm engine.
include("./RCG/engine.jl")

# user-facing.
include("frontend.jl")

export batchfit, runRCG!

end # module VariableMetricCG
