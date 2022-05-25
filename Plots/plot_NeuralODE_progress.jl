using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, DelimitedFiles, PGFPlotsX
#pgfplotsx()

u0 = Float32[1.0; 0.0] # Initial condition
datasize = 100 # Number of data points
tspan = (0.0f0, 3.0f0) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps for each data point

# Function that will generate the data we are trying to fit
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 5.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)' # Need transposes to make the matrix multiplication work
end

Data_raw = readdlm("intermediate_results_2wave.txt")
n = 100;

# Define the problem with the function above
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Solve and take just the solution array
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

k1 = 1;
data1 = Data_raw[k1*100+1:k1+n]
#scatter(tsteps,data1,color="lightblue1",markerstrokewidth=0)
k2 = 100;
data2 = Data_raw[k2*100+1:k2*100+n]
scatter(tsteps,data2,color="lightblue1",markerstrokewidth=0,size=(1000,300),dpi=1000,legend=:bottomright,label="100 iterations")
k3 = 300;
data3 = Data_raw[k3*100+1:k3*100+n]
scatter!(tsteps,data3,color="deepskyblue",markerstrokewidth=0,label="200 iterations")
k4 = 303;
data4 = Data_raw[k4*100+1:k4*100+n]
scatter!(tsteps,data4,color="dodgerblue2",markerstrokewidth=0,label="300 iterations")
k5 = 305;
data5 = Data_raw[k5*100+1:k5*100+n]
scatter!(tsteps,data5,color="blue2",markerstrokewidth=0,label="400 iterations")
k6 = 400;
data6 = Data_raw[k6*100+1:k6*100+n]
scatter!(tsteps,data6,color="navy",markerstrokewidth=0,label="500 iterations")
scatter!(tsteps,ode_data[1,:],color="black",markerstrokewidth=0,label="data")
#savefig( "NeuralODE_progress.png")