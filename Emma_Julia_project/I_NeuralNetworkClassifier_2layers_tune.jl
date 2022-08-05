include("../Data_Handling_NN.jl")
using MLJFlux, Flux, OpenML, MLJModelInterface

##2 layers

abstract type Builder <: MLJModelInterface.MLJType end

mutable struct Builder_I <: Builder
	n1::Int
    n2::Int
    dropout::Float64
    σ
end

Builder_I(; n1=1, n2=1, dropout=0.5, σ=Flux.relu) = Builder_I(n1, n2, dropout, σ)

function MLJFlux.build(builder::Builder_I, rng, n_in, n_out)
	init = Flux.glorot_uniform(rng)
    return Flux.Chain(
                Flux.Dense(n_in, builder.n1, builder.σ, init=init),
                Flux.Dropout(builder.dropout),
                Flux.Dense(builder.n1, builder.n2, builder.σ, init=init),
                Flux.Dropout(builder.dropout),
                Flux.Dense(builder.n2, n_out, builder.σ, init=init))
	
end

Random.seed!(31)
model_Ia = NeuralNetworkClassifier(builder = Builder_I(σ=Flux.relu), optimiser = ADAM(), batch_size = 30, epochs = 50)
self_tuning_model_Ia = TunedModel(model = model_Ia,
                                resampling = CV(nfolds= 5, shuffle = true, rng = 31),
                                range = [range(model_Ia,
                                            :(builder.n1),
                                            values = [20, 30, 40]),
                                        range(model_Ia,
                                            :(builder.n2),
                                            values = [20, 30, 40]),
                                        range(model_Ia,
                                            :(builder.dropout),
                                            values = [0.1, 0.2, 0.5, 0.7])],
                                measure = auc,
                                acceleration = CPUProcesses(),
                                acceleration_resampling = CPUThreads())
mach_Ia = fit!(machine(self_tuning_model_Ia, trainingSet_median_std, precipitationNextday))

plot(mach_Ia)

report(mach_Ia).best_model
n1_Ia = report(mach_Ia).best_model.builder.n1
n2_Ia = report(mach_Ia).best_model.builder.n2
report(mach_Ia).best_model.builder.dropout
report(mach_Ia).best_history_entry.measurement

#5, 5, d = 0.2 -->  0.9056457003434227
#7, 5, d= 0.5 -->   0.919589523441722
#10, 10, d = 0.7 --> 0.9222002822712001
#15, 12, d = 0.7 --> 0.92357912457595
#30, 20, d = 0.7 --> 0.9243648939370201

Random.seed!(31)
model_Ib = NeuralNetworkClassifier(builder = Builder_I(n1 = n1_Ia, n2= n2_Ia, σ=Flux.relu), optimiser = ADAM(), batch_size = 30)
self_tuning_model_Ib = TunedModel(model = model_Ib,
                                resampling = CV(nfolds= 20),
                                tuning = Grid(goal = 20),
                                range = [range(model_Ib,
                                            :(builder.dropout),
                                            lower = 0.62,
                                            upper = 0.67),
                                        range(model_Ib,
                                            :epochs,
                                            lower = 35,
                                            upper = 40)],
                                measure = auc,
                                acceleration = CPUProcesses(),
                                acceleration_resampling = CPUThreads())
mach_Ib = fit!(machine(self_tuning_model_Ib, trainingSet_median_std, precipitationNextday))

plot(mach_Ib)

report(mach_Ib).best_model
d_Ib = report(mach_Ib).best_model.builder.dropout
epochs_Ib = report(mach_Ib).best_model.epochs
report(mach_Ib).best_history_entry.measurement

#5, 7, d = 0.5631578947368421 --> 0.918216010484387
#15, 12, d = 0.6789473684210526 --> 0.9263346793852019
#30, 20, d= 0.7, epochs = 35 --> 0.9263921913613423
#30, 20, d= 0.65, epochs = 37 --> 0.9267172814819047
#30, 20, d = 0.67, epochs = 38 --> 0.926654433186483


Random.seed!(31)
machall_Ib2 = fit!(machine(NeuralNetworkClassifier(builder = Builder_I(n1 = n1_Ia, n2 = n1_Ia, σ=Flux.relu, dropout = d_Ib), optimiser = ADAM(), batch_size = 30, epochs = epochs_Ib), trainingSet_median_std, precipitationNextday))

prediction_I2 = predict_mode(machall_Ib2, testSet_median_std) 
pred_proba_I2 = predict(machall_Ib2, testSet_median_std)

sample_submission_I2 = DataFrame(id = [1:length(pred_proba_I2);], precipitation_nextday = pdf(pred_proba_I2, levels(prediction_I2))[:,2])
CSV.write("sample_submissionI2_NeuralNetworkClassifier_2Layers_Tuned.csv", sample_submission_I2)