require 'nn';
require 'torch';
dofile 'A3_skeleton.lua'

-- Script sets up a neural net that only pools using
-- provided pooling algorithm.


-- TEST SCRIPT FOR THE ABOVE FUNCTIONS
ninputs = 10
batch_size = 2

x = torch.rand(ninputs, batch_size)
gradOutput = torch.ones(4):div(2)
print("Input tensor: ")
print(x)

-- If TemporalLogExpPooling is added to a nn.Sequential() model, it breaks on :add() function
model = nn.Sequential()
model:add(nn.TemporalLogExpPooling(3,1,.5))
-- model:add(nn.TemporalMaxPooling(3,2))

-- If TemporalLogExpPooling is used without nn.Sequential container, it seems to work
-- model = nn.TemporalLogExpPooling(3, 2, .5)

-- Feed forward
model_out = model:forward(x)
print("Model output: ")
print(model_out)

-- Define fake gradOutput to feed backward
gradOutput = torch.ones(model_out:size()):div(2)
print("gradOutput")
print(gradOutput)

-- Feed backward
gradInput = model:backward(x, gradOutput)
print("Model gradInput: ")
print(gradInput)


-- USING MAX POOLING
model_max_pooling = nn.TemporalMaxPooling(3, 2)

model_max_pooling_out = model_max_pooling:forward(x)
print("Max pooling output: ")
print(model_max_pooling_out)
grad_input_max_pooling = model_max_pooling:backward(x, model_max_pooling_out)


--------------------------------------------------------------------
-- SOME TESTS

print(">>>>>>>>>>SOME TESTS")
print(">> TEST 1")
x = torch.Tensor(3,1)
x[1] = 1
x[2] = 2
x[3] = 3
beta = 1
output = torch.Tensor(2,1)
gradOutput = torch.Tensor(2,1)
gradInput = torch.Tensor(3,1)
gradOutput[1] = 0.5
gradOutput[2] = 0.2

-- Compute output 'by hand'
exp_beta_x = x:clone():mul(beta):exp()
output[1] = torch.log((exp_beta_x[1] + exp_beta_x[2])*1/2) * 1/beta
output[2] = torch.log((exp_beta_x[2] + exp_beta_x[3])*1/2) * 1/beta
print("Manually computed output: ")
print(output)

-- Compute output using module
model = nn.TemporalLogExpPooling(2,1,beta)
print("TemporalLogExpPooling forward output: ")
print(model:forward(x))

gradInput[1] = exp_beta_x[1] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1]
gradInput[2] = exp_beta_x[2] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1] + exp_beta_x[2] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]
gradInput[3] = exp_beta_x[3] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]

print("Manually computed gradInput")
print(gradInput)
print("TemporalLogExpPooling backward output: ")
print(model:backward(x, gradOutput))

print(">> TEST 2")
x = torch.Tensor(4,2)
x[1][1] = 1
x[2][1] = 2
x[3][1] = 3
x[4][1] = 4
x[1][2] = 4
x[2][2] = 3
x[3][2] = 2
x[4][2] = 1
beta = .5
N = 2
step = 1
output = torch.Tensor(3,2)
gradOutput = torch.Tensor(3,2)
gradInput = torch.Tensor(4,2)
gradOutput[1][1] = 0.5
gradOutput[2][1] = 0.4
gradOutput[3][1] = 0.3
gradOutput[1][2] = 0.6
gradOutput[2][2] = 0.7
gradOutput[3][2] = 0.8


-- Compute output 'by hand'
exp_beta_x = x:clone():mul(beta):exp()
-- print("x:")
-- print(x)
-- print("exp_beta_x")
-- print(exp_beta_x)
output[1][1] = torch.log((exp_beta_x[1][1] + exp_beta_x[2][1])*1/N) * 1/beta
output[2][1] = torch.log((exp_beta_x[2][1] + exp_beta_x[3][1])*1/N) * 1/beta
output[3][1] = torch.log((exp_beta_x[3][1] + exp_beta_x[4][1])*1/N) * 1/beta
output[1][2] = torch.log((exp_beta_x[1][2] + exp_beta_x[2][2])*1/N) * 1/beta
output[2][2] = torch.log((exp_beta_x[2][2] + exp_beta_x[3][2])*1/N) * 1/beta
output[3][2] = torch.log((exp_beta_x[3][2] + exp_beta_x[4][2])*1/N) * 1/beta
print("Manually computed output: ")
print(output)

-- Compute output using module
model = nn.TemporalLogExpPooling(N,step,beta)
print("TemporalLogExpPooling forward output: ")
print(model:forward(x))

-- gradInput[1] = exp_beta_x[1] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1]
-- gradInput[2] = exp_beta_x[2] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1] + exp_beta_x[2] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]
-- gradInput[3] = exp_beta_x[3] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]

-- print("Manually computed gradInput")
-- print(gradInput)
-- print("TemporalLogExpPooling backward output: ")
-- print(model:backward(x, gradOutput))