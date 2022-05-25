using DiffEqFlux, DifferentialEquations, Plots, DelimitedFiles

N = 10^7
dat = readdlm("covid.txt",',');
time_i = dat[:,1]; # times
Infected = 50*dat[:,2]/N; # Normalize data
Infected[2:end] = Infected[2:end];
A = Infected;

tshort_inds = (Int.(time_i)*10).+1
tshort = Int.(time_i).+1;

#tsteps = range(tspan[1], tspan[2], length = datasize)
tspan = (0.0, maximum(time_i))
tsteps = time_i
#=
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
ode_data = ode_data[1,:]
=#
ode_data = Infected
u0 = Float32[0.0; 0.0; 0.0; 0.0]


dudt2 = FastChain(FastDense(4, 100, relu),
                FastDense(100, 50, relu),
                FastDense(50, 4))
prob_neuralode = NeuralODE(dudt2, tspan, Vern7(), saveat = tsteps, abstol=1e-6, reltol=1e-6)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    global tsteps
    pred = predict_neuralode(p)
    k = size(pred,2)
    t_loc = Int.(tsteps[tsteps .<= k] .+1)
    #lloss = sum(abs2, (ode_data[1:size(pred,2)] .- pred[1,:]))
    loss = sum(abs2, (ode_data[1:length(t_loc)] .- pred[1,t_loc]))
    return loss, pred
end

iter = 0
callback = function (p, l, pred; doplot = true)
  global iter
  global tsteps
  iter += 1
  display(l)
  if doplot
    # plot current prediction against data
    k = size(pred,2)
    t_local = tsteps[tsteps .<= k]
    plt = scatter(t_local, ode_data[1:length(t_local)], label = "data")
    scatter!(plt, 0:k-1, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end

t4 = 60.0
prob_neuralode = NeuralODE(dudt2, (0.0,t4), Tsit5(), saveat = 0:t4)
result_neuralode5 = DiffEqFlux.sciml_train(loss_neuralode,
                                           prob_neuralode.p,
                                           ADAM(), maxiters = 1000,
                                           cb = callback)

t6 = 103.0
prob_neuralode = NeuralODE(dudt2, (0.0,t6), Tsit5(), saveat = 0:t6)
result_neuralode7 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode6.u,
                                           ADAM(), maxiters = 1000,
                                           cb = callback)

t7 = 134.0
prob_neuralode = NeuralODE(dudt2, (0.0,t7), Tsit5(), saveat = 0:t7)
result_neuralode8 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode7.u,
                                           ADAM(), maxiters = 1000,
                                           cb = callback)

t8 = 149.0
prob_neuralode = NeuralODE(dudt2, (0.0,t8), Tsit5(), saveat = 0:t8)
result_neuralode9 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode8.u,
                                           ADAM(), maxiters = 1000,
                                           cb = callback)

t9 = 170.0
prob_neuralode = NeuralODE(dudt2, (0.0,t9), Tsit5(), saveat = 0:t9)
result_neuralode10 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode9.u,
                                           ADAM(), maxiters = 1000,
                                           cb = callback)                                          