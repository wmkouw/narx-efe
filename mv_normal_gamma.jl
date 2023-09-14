module mv_normal_gamma

using LinearAlgebra
using Distributions
using SpecialFunctions

export MvNormalGamma, pdf, logpdf, params, dimensions

mutable struct MvNormalGamma <: ContinuousMultivariateDistribution
 
    D ::Integer
    μ ::Vector
    Λ ::Matrix
    α ::Real
    β ::Real

    function MvNormalGamma(mean::Vector, precision_matrix::Matrix, shape::Float64, rate::Float64)
        
        if shape <= 0.0; error("Shape parameter must be positive."); end
        if rate <= 0.0;  error("Rate parameter must be positive."); end
        
        dimensions = length(mean)
        if size(precision_matrix, 1) != dimensions
            error("Number of rows of precision matrix does not match mean vector length.")
        end
        if size(precision_matrix, 2) != dimensions
            error("Number of columns of precision matrix does not match mean vector length.")
        end

        return new(dimensions, mean, precision_matrix, shape, rate)
    end
end

function dims(p::MvNormalGamma)
    return p.D
end

function params(p::MvNormalGamma)
    return p.μ, p.Λ, p.α, p.β
end

function pdf(p::MvNormalGamma, θ, τ)
    μ, Λ, α, β = params(p)
    return det(Λ)^(1/2) * (2π)^(-p.D/2)*β^α/gamma(α)*τ^(α+p.D/2-1)*exp( -τ/2*((θ-μ)'*Λ*(θ-μ) +2β) )
end

function logpdf(p::MvNormalGamma, θ, τ)
    μ, Λ, α, β = params(p)
    return 1/2*logdet(Λ) -p.D/2*log(2π) + α*log(β) - log(gamma(α)) +(α+p.D/2-1)*log(τ) -τ/2*((θ-μ)'*Λ*(θ-μ) +2β)
end

end