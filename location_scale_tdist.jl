module location_scale_tdist

using LinearAlgebra
using Distributions
using SpecialFunctions

export LocationScaleT, pdf, logpdf, params, dimensions

mutable struct LocationScaleT <: ContinuousUnivariateDistribution
 
    ν ::Real
    μ ::Real
    σ ::Real

    function LocationScaleT(ν::Float64, μ::Float64, σ::Float64)
        
        if ν <= 0.0; error("Degrees of freedom parameter must be positive."); end
        if σ <= 0.0; error("Scale parameter must be positive."); end

        return new(ν, μ, σ)
    end
end

function params(p::LocationScaleT)
    return p.ν, p.μ, p.σ
end

function pdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return gamma( (ν+1)/2 ) / ( gamma(ν/2) *sqrt(πν)*σ ) * ( 1 + (x-μ)^2/(ν*σ^2) )^( -(ν+1)/2 )
end

function logpdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return loggamma( (ν+1)/2 ) - loggamma(ν/2) - 1/2*log(πν) - log(σ) + ( -(ν+1)/2 )*log( 1 + (x-μ)^2/(ν*σ^2) )
end

end