using DiffEqFlux, DifferentialEquations, Plots


u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 16, tanh),
                  FastDense(16, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Vern7(), saveat = tsteps, abstol=1e-6, reltol=1e-6)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, (ode_data[:,1:size(pred,2)] .- pred))
    return loss, pred
end

iter = 0
callback = function (p, l, pred; doplot = true)
  global iter
  iter += 1

  display(l)
  if doplot
    # plot current prediction against data
    plt = scatter(tsteps[1:size(pred,2)], ode_data[1,1:size(pred,2)], label = "data")
    scatter!(plt, tsteps[1:size(pred,2)], pred[1,:], label = "prediction")
    display(plot(plt))
  end

  return false
end


prob_neuralode = NeuralODE(dudt2, (0.0,1.5), Tsit5(), saveat = tsteps[tsteps .<= 1.5])
result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,ADAM(0.05), cb = callback,maxiters = 300)
callback(result_neuralode2.u,loss_neuralode(result_neuralode2.u)...;doplot=true)