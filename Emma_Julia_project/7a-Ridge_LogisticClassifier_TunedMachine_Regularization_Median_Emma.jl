#= 
	Ridge Regression on a multiple Linear Classification == LogisticClassifier
	Self Tuning Machine to find the best lambda for the Ridge Regression
	Fill in the missing values with the median of all values rounded to the next integer and with the most occurent prediction (true or false) using FillImputer
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
using CSV, DataFrames, Plots, MLJ, MLJLinearModels, Distributions

#Load the dataSet
trainingdata = CSV.read(joinpath(@__DIR__, "trainingdata.csv"), DataFrame)
testSet = CSV.read(joinpath(@__DIR__, "testdata.csv"), DataFrame)

#Extract the dataSet 
trainingSet7 = DataFrame(select(trainingdata, Not([:precipitation_nextday, :ALT_sunshine_4])))
precipitationNextday7 = coerce(trainingdata.precipitation_nextday, Binary)

#Handle the missing values
trainingSet7 = MLJ.transform(fit!(machine(FillImputer(), trainingSet7)), trainingSet7)
testSet = select(testSet, Not([:ALT_sunshine_4]))

#Standardization
standardMach7 = fit!(machine(Standardizer(), trainingSet7));
trainingSet7s = MLJ.transform(standardMach7, trainingSet7)

testSets = MLJ.transform(standardMach7, testSet)

#Implement a self tuning model and fit machine
model7 = LogisticClassifier(penalty = :l2)
selfTuningModel7 = TunedModel(model = model7, 
							  resampling = CV(nfolds = 10), 
							  tuning = Grid(goal = 50),
							  range = range(model7, :lambda, lower = 125, upper = 140), 
							  measure = auc)

mach7 = machine(selfTuningModel7, trainingSet7s, precipitationNextday7) |> fit!

#Estimate the expected test error
report(mach7)
report(mach7).best_model
report(mach7).best_model.lambda
auc7 = report(mach7).best_history_entry.measurement #auc = 0.9260376882241121

#Plot the error in function of lambda
plot(mach7)

#Fit the best model on all the available data 
mach7all = machine(LogisticClassifier(penalty = :l2, lambda = report(mach7).best_model.lambda), trainingSet7s, precipitationNextday7) |> fit!

#Make predictions
prediction7 = predict_mode(mach7all, testSets) 
pred_proba7 = predict(mach7all, testSets)

#Save the results
sample_submission7 = DataFrame(id = [1:length(pred_proba7);], precipitation_nextday = pdf(pred_proba7, levels(prediction7))[:,2])
CSV.write("sample_submission7a_linearMethod_TunedMachine_RidgeRegressor_standard_median.csv", sample_submission7)