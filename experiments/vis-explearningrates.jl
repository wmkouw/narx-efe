using Revise
using JLD
using LaTeXStrings
using Plots
default(label="", margin=20Plots.pt)
includet("../mv_normal_gamma.jl"); using .mv_normal_gamma

num_exps = 100
N = 100

id_rnd = zeros(num_exps, N)
id_MSE = zeros(num_exps, N)
id_EFE = zeros(num_exps, N)
id_sin = zeros(num_exps, N)

for nn in 1:num_exps

    DD = load("./experiments/results/learningrate-rnd-$nn.jld")
    id_rnd[nn,:] = [mv_normal_gamma.logpdf(MvNormalGamma(DD["mu"][k], DD["Lambda"][k], DD["alpha"][k], DD["beta"][k]), DD["sys_theta"], inv(DD["sys_sd"]^2)) for k in 1:N]
    
    DD = load("./experiments/results/learningrate-MSE-$nn.jld")
    id_MSE[nn,:] = [mv_normal_gamma.logpdf(MvNormalGamma(DD["mu"][k], DD["Lambda"][k], DD["alpha"][k], DD["beta"][k]), DD["sys_theta"], inv(DD["sys_sd"]^2)) for k in 1:N]

    DD = load("./experiments/results/learningrate-EFE-$nn.jld")
    id_EFE[nn,:] = [mv_normal_gamma.logpdf(MvNormalGamma(DD["mu"][k], DD["Lambda"][k], DD["alpha"][k], DD["beta"][k]), DD["sys_theta"], inv(DD["sys_sd"]^2)) for k in 1:N]
    
    DD = load("./experiments/results/learningrate-sin-$nn.jld")
    id_sin[nn,:] = [mv_normal_gamma.logpdf(MvNormalGamma(DD["mu"][k], DD["Lambda"][k], DD["alpha"][k], DD["beta"][k]), DD["sys_theta"], inv(DD["sys_sd"]^2)) for k in 1:N]
end

plot(xlabel="time (#steps)", grid=true, guidefontsize=18, tickfontsize=15, legendfontsize=15, ylabel=L"$ \ln \, p(θ_{*}, \tau_{*} \, | \, \mu, \Lambda, \alpha, \beta)$", size=(900,400), legend=:topleft)
# plot(mean(id_rnd, dims=1)', ribbon=std(id_rnd, dims=1)', label="rnd")
plot!(mean(id_rnd, dims=1)', label="rnd")
plot!(mean(id_sin, dims=1)', label="sin")
plot!(mean(id_MSE, dims=1)', label="MSE")
plot!(mean(id_EFE, dims=1)', label="EFE")