using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, CUDA, Plots, DelimitedFiles
CUDA.allowscalar(false)


function node_model(t)
    dudt2 = FastChain(FastDense(4, 100, swish),
            FastDense(100, 50, swish),
            FastDense(50, 4))
    node = NeuralODE(dudt2, (minimum(t), maximum(t)), 
             Tsit5(), saveat = t, 
             abstol = 1e-9, reltol = 1e-9)
end

function plot_pred(t, u, û)
    plt = scatter(reshape(t,:), reshape(u,:), 
          label = "Data")
    scatter!(plt, reshape(t,:), reshape(û,:), 
         label = "Prediction")
end

function train(node, t, u, p = nothing; maxiters, lr)
    u0 = [u[1:1],u[1:1],u[1:1],u[1:1]]
    predict(p) = Array(node(u0, p))

    loss(p) = begin
        û = predict(p)[1,:]
        l = sum(abs2,vec(û).-vec(u))
    return l, û
    end

    losses = []
    cb(p, l, û) = begin
    push!(losses, l)
    false
    end

    p = p == nothing ? node.p : p
    res = DiffEqFlux.sciml_train(loss, p, ADAMW(lr),
                 maxiters = maxiters,
                 cb = cb)

    p_pred = plot_pred(t, u, predict(res.minimizer))
    plot!(p_pred, legend = (.11, .9))
    p_loss = plot(losses, label="Loss")  
    display(plot(p_pred, p_loss, layout=(2, 1)))
    return res.minimizer
end

#=
function node_model(t)
    dudt2 = FastChain(FastDense(2, 100, swish),
          FastDense(100, 2))
  
    node = AugmentedNDELayer(NeuralODE(dudt2, (minimum(t), maximum(t)),
             Tsit5(), saveat = t,
             abstol = 1e-12, reltol = 1e-12), 1)
  end
  
function plot_pred(t, u, û)
    plt = scatter(reshape(t,:), reshape(u,:),
          label = "Data")
    scatter!(plt, reshape(t,:), reshape(û,:),
         label = "Prediction")
end
  
function train(node, t, u, p = nothing; maxiters, optimizer, kwargs...)
    u0 = u[1:1]
    predict(p) = Array(node(u0, p))
  
    loss(p) = begin
        û = predict(p)[1,:]
        l = sum(abs2,vec(û)-vec(u))
        return l, û
    end
  
    losses = []
    cb(p, l, û) = begin
        @show l
        push!(losses, l)
        false
    end
  
    p = p == nothing ? node.p : p
    res = DiffEqFlux.sciml_train(loss, p, optimizer;
                 maxiters = maxiters,
                 cb = cb,
                 kwargs...)
  
    p_pred = plot_pred(t, u, predict(res.minimizer)[1,:])
    plot!(p_pred, legend = (.11, .9))
    p_loss = plot(losses, label="Loss")
    display(plot(p_pred, p_loss, layout=(2, 1)))
    return res.minimizer
  end
=#

N = 10^7
dat = readdlm("covid.txt",',');
time_i = dat[:,1]; # times
Infected = 50*dat[:,2]/N; # Normalize data

# train on the k first observations
k=5
node = node_model(time_i)
p = train(node, time_i, Infected, maxiters = 100, lr = .02);